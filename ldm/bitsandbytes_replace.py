import bitsandbytes as bnb
import torch.nn as nn
import torch

# Taken from https://github.com/hpcaitech/ColossalAI/blob/main/examples/images/diffusion/scripts/utils.py#L5
class Linear8bit(nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        has_fp16_weights=False,
        memory_efficient_backward=False,
        threshold=6.0,
        weight=None,
        bias=None
    ):
        super(Linear8bit, self).__init__(
            input_features, output_features, bias is not None
        )
        self.state = bnb.MatmulLtState()
        self.bias = bias
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True
            
        self.register_parameter("SCB", nn.Parameter(torch.empty(0), requires_grad=False))
        self.weight = weight
        self.quant()
        

    def quant(self):  
        weight = self.weight.data.contiguous().half().cuda()
        CB, _, SCB, _, _ = bnb.functional.double_quant(weight)
        delattr(self, "weight")
        setattr(self, "weight", nn.Parameter(CB, requires_grad=False))
        delattr(self, "SCB")
        setattr(self, "SCB", nn.Parameter(SCB, requires_grad=False))
        del weight

    def forward(self, x):
        self.state.is_training = self.training
        
        if self.bias is not None and self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.half()
        
        self.state.CB = self.weight.data
        self.state.SCB = self.SCB.data
        
        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
        del self.state.CxB
        return out

def replace_module(model):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_module(module)

        if isinstance(module, nn.Linear) and "out_proj" not in name:    
            linear_8 = Linear8bit(
                input_features=module.in_features,
                output_features=module.out_features,
                threshold=6.0,
                weight=module.weight,
                bias=module.bias,
            )
            print(f"**** found and replaced {name} w. {Linear8bit}")
            
            setattr(model, name, linear_8)
    return model


def bitsandbytes_injection(module):
    for _, module in module.named_children():
        if len(list(module.children())) > 0:
            replace_module(module)