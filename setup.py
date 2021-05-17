#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup


requirements = ["torch", "torchvision"]


setup(
    name="visionlongformer4od",
    version="0.1",
    author="PengchuanZhang",
    url="https://github.com/microsoft/VisionLongformerForObjectDetection",
    description="Vision Longformer For Object Detection",
    packages=find_packages(exclude=("configs", "tests",)),
    install_requires=["einops", "dataclasses", "shapely", "timm", "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"],
)
