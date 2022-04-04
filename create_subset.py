# Copyright (c) 2022-present, Enoch Tetteh, Inc.
# All rights reserved.

import os
import shutil
import splitfolders
from glob import glob

def split_train_val_test(src_path: str, dst_path: str, split_ratio: tuple):
    """
    The split_train_val_test function splits a multi-class images directory into
    train, validation, and test sub-folders according to a given ratio.

    Args:
        src_path: source directory of multi-class images.
        dst_path: output directory where dataset splits are saved.
        split_ratio tuple[float, float, float]: ratio of train, val, test splits. (0.70, 0.15, 0.15) represents
                     70% train set, 15% valid set and 15% test set.
    """

    splitfolders.ratio(src_path, output=dst_path, seed=19900710, ratio=split_ratio)


def create_data_subset(dataset_path: str, subset_percent: int):
    """
    This function creates a subset from a given train dataset based on a given percentage,
    and saves the names of the images in the subset as a text file.

    Args:
       dataset_path: path to train dataset
       subset_percent: percentage of train dataset to create a subset
    """

    # split percentages
    rem = 100 - subset_percent
    train_ratio, val_ratio, test_ratio = subset_percent/100, (rem/2)/100, (rem/2)/100
    ratio = (train_ratio, val_ratio, test_ratio)

    src_path = os.path.join(dataset_path, "train")
    dst_path = os.path.join(dataset_path, "subset")

    # split train file to obtain the required subset percentage
    split_train_val_test(src_path=src_path, dst_path=dst_path, split_ratio=ratio)

    # remove created val and test folders. We only need the train split
    shutil.rmtree(os.path.join(dst_path, "val"))
    shutil.rmtree(os.path.join(dst_path, "test"))

    data = os.path.join(dst_path, "train/")
    images_list = []

    # walk through each subfolder, and copy the image name to a list
    for subfolder in os.listdir(data):
        for image in glob(data+subfolder+"/*"):
            images_list.append(os.path.basename(image))

    # save images names in the list to .txt file
    with open(os.path.join(dataset_path, f'{subset_percent}percent_train_subset.txt'), 'w') as f:
        for item in images_list:
            f.write(item + "\n")

    # remove the subset folder
    shutil.rmtree(dst_path)
