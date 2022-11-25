import os
from setuptools import setup, find_packages
from pkg_resources import parse_requirements

_PATH_ROOT = os.path.realpath(os.path.dirname(__file__))
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements.txt")

# load requirements
with open(_PATH_REQUIRE) as fp:
    requirements = list(map(str, parse_requirements(fp.readline())))

setup(
    name='stable-diffusion',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
)