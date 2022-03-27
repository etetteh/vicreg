# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from PIL import ImageOps, ImageFilter
import timm
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment
from torchvision.transforms.functional import InterpolationMode


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TrainTransform(object):
    def __init__(self, train_crop):
        self.train_crop = train_crop
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.train_crop, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.train_crop, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD
                ),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2


class StrongTrainTransform(object):
    def __init__(self, train_crop):
        self.train_crop = train_crop
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.train_crop, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            autoaugment.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                    mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD
                ),
            ]
        )

        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(self.train_crop, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            autoaugment.RandAugment(interpolation=InterpolationMode.BILINEAR),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                    mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD
                ),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2
