# Copyright (c) 2022-present, Enoch Tetteh, Inc.
# All rights reserved.

import torch

def deit(arch: str = 'deit_tiny_patch16_224') -> tuple:
    """
    This function takes a DEIT model
    (https://arxiv.org/abs/2012.12877?fbclid=IwAR1bq7LJA4YII7QEwMXjTKkdhiZ_yzwXmx929Q5X2W4svn0A0gobWu6-V1c)
    and extracts its backbone (non-fully connected layers), and embedding size (input size of fully connected layer)
    Args:
        arch: str architecture name of a DEIT model
    Returns:
        a tuple of model backbone and embedding size
    """

    model = torch.hub.load('facebookresearch/deit:main', model=arch, pretrained=False)
    backbone = torch.nn.Sequential(
            *list(model.children())[:-1]
        )
    embedding = model.head.in_features

    return backbone, embedding
