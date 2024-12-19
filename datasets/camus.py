import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandSpatialCropd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    RandFlipd,
    Resized,
    EnsureTyped
)
from glob import glob
import skimage.io as io

RANDOM_SEED = 123

class DataLoaderCamus(Dataset):
    def __init__(self, dataset_path, input_name, target_name, condition_name, stage, single_frame=True,
                 img_res=(272, 272), img_crop=(256, 256), seg_parts=True, train_ratio=1.0, valid_ratio=0.2, zero_shot=False):
        self.dataset_path = dataset_path
        self.input_name = input_name
        self.target_name = target_name
        self.condition_name = condition_name
        self.spatial_size = img_res[0]
        self.crop_size = img_crop[0]
        self.single_frame = single_frame
        self.seg_parts = seg_parts
        self.is_train = stage == 'train'

        # Initialize transformations
        self.transform = self.get_transform(self.is_train)

        self.fnames = []

        # Load patients
        patients = []
        if zero_shot:
            for dir in sorted(glob(os.path.join(self.dataset_path, 'testing/image', '*'))):
                if os.listdir(dir):  # Only include non-empty directories
                    patients.append(dir)
            self.data_list = patients
            print(f"#test: {len(self.data_list)}")
        else:
            for dir in sorted(glob(os.path.join(self.dataset_path, 'training', '*'))):
                if os.listdir(dir):
                    patients.append(dir)

            random.seed(RANDOM_SEED)
            random.shuffle(patients)

            num_patients = len(patients)
            num_train = int(num_patients * train_ratio)
            num_valid = int(num_train * valid_ratio)

            self.train_patients = patients[num_valid:num_train]
            self.valid_patients = patients[:num_valid // 2]
            self.test_patients = patients[num_valid // 2:num_valid]

            if stage == 'train':
                self.data_list = self.train_patients
                print(f"#train: {len(self.data_list)}")
            elif stage == 'valid':
                self.data_list = self.valid_patients
                print(f"#valid: {len(self.data_list)}")
            elif stage == 'test':
                self.data_list = self.test_patients
                print(f"#test: {len(self.data_list)}")

        self.data_length = len(self.data_list)
        self.fnames = patients

    def __getitem__(self, index):
        path = self.data_list[index]
        input_path, condition_path = self.get_path(path)
        input_img = self.read_image(input_path, '_gt' in self.input_name)
        condition_img = self.read_image(condition_path, True)
        if self.seg_parts:
            LV = np.where(condition_img == 1, 1, 0)
            LA = np.where(condition_img == 3, 0, 0)
            condition_img = np.stack([LV], axis=0)

        input_dict = self.transform({'images': input_img, 'masks': condition_img})
        return input_dict['images'] / 255.0, input_dict['masks'] / 1.0, index


    def __len__(self):
        return len(self.data_list)

    def read_image(self, img_path, is_gt):
        """Reads and returns an image."""
        return io.imread(img_path, plugin='simpleitk').squeeze()

    def get_path(self, input_folder):
        """Generates paths for input and condition images."""
        for root, _, files in os.walk(input_folder):
            for label_key in ["_ED", "_half_sequence", "_ES"]:
                for file_name in sorted(files):
                    if label_key in file_name and "_4CH_" in file_name and file_name.endswith(".nii.gz"):
                        input_path = os.path.join(root, file_name)
                        condition_path = input_path.replace("image", "mask").replace(".nii.gz", "_gt.nii.gz")
                        return input_path, condition_path
        return None, None

    def get_transform(self, is_train):
        """Sets up transformations."""
        all_keys = ['images', 'masks']
        spatial_size = (self.spatial_size, self.spatial_size)
        crop_size = (self.crop_size, self.crop_size)

        if is_train:
            transform = Compose([
                AddChanneld(keys=['images'] if self.seg_parts else all_keys, allow_missing_keys=True),
                Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True, mode='nearest'),
                RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1),
                EnsureTyped(keys=all_keys, allow_missing_keys=True),
            ])
        else:
            transform = Compose([
                AddChanneld(keys=['images'] if self.seg_parts else all_keys, allow_missing_keys=True),
                Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True, mode='nearest'),
                CenterSpatialCropd(keys=all_keys, roi_size=crop_size, allow_missing_keys=True),
                EnsureTyped(keys=all_keys, allow_missing_keys=True),
            ])
        return transform


if __name__ == '__main__':
    data_loader = DataLoaderCamus(
        dataset_path='/home/bassant/dataset/kaggle/old/all_views',
        input_name="4CH_ED",
        target_name="4CH_ED",
        condition_name="4CH_ED_gt",
        stage="test",
        zero_shot=True
    )
    from monai.data import DataLoader
    train_loader = DataLoader(data_loader, batch_size=2, shuffle=False, num_workers=1)

    for targets, targets_gt, _ in train_loader:
        print(targets.shape)
        print(targets_gt.shape)
