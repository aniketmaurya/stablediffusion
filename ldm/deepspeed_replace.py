import torch
from functools import partial
from dataclasses import dataclass
import deepspeed.ops.transformer as transformer_inference
from deepspeed.ops.transformer.inference.diffusers_attention import DeepSpeedDiffusersAttention
from deepspeed.ops.transformer.inference.diffusers_transformer_block import DeepSpeedDiffusersTransformerBlock
from deepspeed.ops.transformer.inference.diffusers_2d_transformer import Diffusers2DTransformerConfig
from ldm.modules.attention import CrossAttention, BasicTransformerBlock
from deepspeed.module_inject.replace_policy import UNetPolicy, DSPolicy
from ldm.models.diffusion.ddpm import DiffusionWrapper
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

@dataclass
class CudaGraphRecord:

    graph = None
    args = None
    kwargs = None
    output = None
    cuda_graph_created: bool = False
    enable_cuda_graph: bool = True


class CudaGraphBatchRecord(dict):

    def __init__(self, enable_cuda_graph):
        super().__init__()
        self.enable_cuda_graph = enable_cuda_graph

    def __getitem__(self, key):
        if key not in self:
            self[key] = CudaGraphRecord(enable_cuda_graph=self.enable_cuda_graph)
        for item, value in self.items():
            if item == key:
                return value
        raise Exception()


class CudaGraphInferenceModule(torch.nn.Module):

    # Inspired from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/model_implementations/diffusers/vae.py

    inference_methods = ["forward"]

    def __init__(self, module, enable_cuda_graph = True):
        super().__init__()
        self.module = module
        self.module.requires_grad_(requires_grad=False)
        self.module.to(memory_format=torch.channels_last)
        self.cuda_graph_records = {}

        for method_name in self.inference_methods:
            fn = getattr(self, f"_{method_name}")
            assert fn
            self.cuda_graph_records[method_name] = CudaGraphBatchRecord(enable_cuda_graph=enable_cuda_graph)
            setattr(self, method_name, partial(self._apply_fn, fn=fn, graph_record=self.cuda_graph_records[method_name]))

    def __getattr__(self, key):
        if hasattr(self._modules["module"], key) and key not in self.__dict__:
            return getattr(self._modules["module"], key)
        if key == "module":
            return self._modules["module"]
        return object.__getattribute__(self, key)

    def _graph_replay(self, graph_record, *args, **kwargs):
        for i in range(len(args)):
            if torch.is_tensor(args[i]):
                graph_record.args[i].copy_(args[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                graph_record.kwargs[k].copy_(kwargs[k])
        graph_record.graph.replay()
        return graph_record.output

    def extract_batch_size(self, *args, **kwargs) -> int:
        raise NotImplementedError

    def _apply_fn(self, *args, fn=None, graph_record=None,  **kwargs):
        batch_size = self.extract_batch_size(*args, **kwargs)
        if graph_record[batch_size].enable_cuda_graph:
            if graph_record[batch_size].cuda_graph_created:
                outputs = self._graph_replay(graph_record[batch_size], *args, **kwargs)
            else:
                self._create_cuda_graph(fn, graph_record[batch_size], *args, **kwargs)
                outputs = self._graph_replay(graph_record[batch_size], *args, **kwargs)
            return outputs
        else:
            return self._forward(*args, **kwargs)

    def _create_cuda_graph(self, fn, graph_record, *args, **kwargs):
        # Warmup to create the workspace and cublas handle
        cuda_stream = torch.cuda.Stream()
        cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cuda_stream):
            for _ in range(3):
                fn(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(cuda_stream)

        # Capture inputs to the graph
        graph_record.graph = torch.cuda.CUDAGraph()
        graph_record.args = args
        graph_record.kwargs = kwargs

        with torch.cuda.graph(graph_record.graph):
            # Store output
            graph_record.output = fn(*graph_record.args, **graph_record.kwargs)

        graph_record.cuda_graph_created = True


class ReplayCudaGraphUnet(CudaGraphInferenceModule):

    def extract_batch_size(self, sample, timestamp, c_crossattn) -> int:
        return sample.shape[0] // 2

    def _forward(self, sample, timestamp, c_crossattn):
        return self.module(sample, timestamp, c_crossattn=c_crossattn)


class ReplayCudaGraphVAE(CudaGraphInferenceModule):

    inference_methods = ["forward", "encode", "decode"]

    def extract_batch_size(self, input, **__) -> int:
        return input.shape[0]

    def _encode(self, x):
        return self.module.encode(x)

    def _decode(self, x):
        return self.module.decode(x)

    def _forward(self, input, sample_posterior=True):
        return self.module(input, sample_posterior=sample_posterior)


class ReplayCudaGraphClipEncoder(CudaGraphInferenceModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module.transformer.text_model._build_causal_attention_mask = self._build_causal_attention_mask

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        mask = torch.empty(bsz,
                           seq_len,
                           seq_len,
                           dtype=dtype,
                           device=torch.cuda.current_device())
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)
        mask = mask.unsqueeze(1)
        return mask

    def extract_batch_size(self, sample, **__) -> int:
        return sample.shape[0]

    def _forward(self, *inputs, **kwargs):
        return self.enc(*inputs, **kwargs)


class UNetPolicy(DSPolicy):

    def match(self, module):
        return isinstance(module, DiffusionWrapper)

    def apply(self, module, enable_cuda_graph=True):
        return ReplayCudaGraphUnet(module, enable_cuda_graph=enable_cuda_graph)

    def attention(self, client_module):
        qw = client_module.to_q.weight
        kw = client_module.to_k.weight
        vw = client_module.to_v.weight

        if qw.shape[1] == kw.shape[1]:
            qkvw = torch.nn.Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)

            return qkvw, \
                   client_module.to_out[0].weight, \
                   client_module.to_out[0].bias, \
                   qw.shape[-1], \
                   client_module.heads
        else:
            return qw, \
                   kw, vw, \
                   client_module.to_out[0].weight, \
                   client_module.to_out[0].bias, \
                   qw.shape[-1], \
                   client_module.heads


class VAEPolicy(DSPolicy):

    def match(self, module):
        return isinstance(module,  AutoencoderKL)

    def apply(self, module, enable_cuda_graph=True):
        return module
        # return ReplayCudaGraphVAE(module, enable_cuda_graph=enable_cuda_graph)


class ClipEncoderPolicy(DSPolicy):

    def match(self, module):
        return isinstance(module, FrozenCLIPEmbedder)

    def apply(self, module, enable_cuda_graph=True):
        return module
        # return ReplayCudaGraphClipEncoder(module, enable_cuda_graph=enable_cuda_graph)


GENERIC_POLICIES = [UNetPolicy, VAEPolicy, ClipEncoderPolicy]


def _module_match(module):
    for policy in GENERIC_POLICIES:
        policy = policy()
        if policy.match(module):
            return policy
    return None

# Inspired from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_module.py#L201
def deepspeed_injection(module, fp16=False, enable_cuda_graph=True):

    def replace_attn(child, policy):
        policy_attn = policy.attention(child)
        if policy_attn is None:
            return child
        if len(policy_attn) == 5:
            qkvw, attn_ow, attn_ob, hidden_size, heads = policy_attn
        else:
            qw, kw, vw, attn_ow, attn_ob, hidden_size, heads = policy_attn

        config = transformer_inference.DeepSpeedInferenceConfig(
            hidden_size=hidden_size,
            heads=heads,
            fp16=fp16,
            triangular_masking=False,
            max_out_tokens=4096,
        )
        attn_module = DeepSpeedDiffusersAttention(config)

        def transpose(data):
            data = data.contiguous()
            data.reshape(-1).copy_(data.transpose(-1, -2).contiguous().reshape(-1))
            data = data.reshape(data.shape[-1], data.shape[-2])
            data.to(torch.cuda.current_device())
            return data

        if len(policy_attn) == 5:
            attn_module.attn_qkvw.data = transpose(qkvw.data)
        else:
            attn_module.attn_qkvw = None
            attn_module.attn_qw.data = transpose(qw.data)
            attn_module.attn_kw.data = transpose(kw.data)
            attn_module.attn_vw.data = transpose(vw.data)

        attn_module.attn_qkvb = None
        attn_module.attn_ow.data = transpose(attn_ow.data)
        attn_module.attn_ob.data.copy_(attn_ob.data.to(torch.cuda.current_device()))
        return attn_module

    def replace_attn_block(child, policy):
        # Track DeepSpeed Issue: https://github.com/microsoft/DeepSpeed/issues/2681
        config = Diffusers2DTransformerConfig(int8_quantization=False)
        return DeepSpeedDiffusersTransformerBlock(child, config)

    new_policies = {
        CrossAttention: replace_attn,
        BasicTransformerBlock: replace_attn_block,
    }

    for name, sub_module in module.named_children():

        policy = _module_match(sub_module)

        if policy is None:
            continue

        def _replace_module(module, policy):
            for name, child in module.named_children():
                _replace_module(child, policy)
                if child.__class__ in new_policies:
                    replaced_module = new_policies[child.__class__](child,
                                                                    policy)
                    setattr(module, name, replaced_module)

        _replace_module(sub_module, policy)
        new_module = policy.apply(sub_module,
                                    enable_cuda_graph=enable_cuda_graph)
        print(f"**** found and replaced {name} w. {type(new_module)}")
        setattr(module, name, new_module)

