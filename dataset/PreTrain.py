import os
import numpy as np
from glob import glob

import torch
from torch.utils import data
import torchvision.transforms as TF

from transforms import transforms as mytrans
import myutils
import dataset
from torchvision.transforms import InterpolationMode
from torchvision import transforms

class PreTrain(data.Dataset):

    def __init__(self, root, output_size, dataset_file='dataset.txt', clip_n=3, max_obj_n=11):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.max_obj_n = max_obj_n
        self.crop = True

        self.img_list = list()
        self.mask_list = list()

        dataset_path = os.path.join(root, dataset_file)
        dataset_list = list()
        with open(os.path.join(dataset_path), 'r') as lines:
            for line in lines:
                dataset_name = line.strip()
                dataset_list.append(dataset_name)

                img_dir = os.path.join(root, 'JPEGImages', dataset_name)
                mask_dir = os.path.join(root, 'Annotations', dataset_name)

                img_list = sorted(glob(os.path.join(img_dir, '*.jpg'))) + sorted(glob(os.path.join(img_dir, '*.png')))
                mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

                assert len(img_list) == len(mask_list)

                self.img_list += img_list
                self.mask_list += mask_list

        # self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)
        # self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.03)
        # self.random_affine = mytrans.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
        # self.random_resize_crop = mytrans.RandomResizedCrop(output_size, (0.8, 1))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0),  # No hue change here as that's not realistic
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9, 1.1), shear=10, interpolation=InterpolationMode.BICUBIC,
                                    fill=(124, 116, 104)),
            transforms.Resize(self.output_size, InterpolationMode.BICUBIC),
            transforms.RandomCrop((self.output_size, self.output_size), pad_if_needed=True, fill=(124, 116, 104)),
        ])

        self.pair_im_dual_transform_nocrop = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9, 1.1), shear=10, interpolation=InterpolationMode.BICUBIC,
                                    fill=(124, 116, 104)),
            transforms.Resize(self.output_size, InterpolationMode.BICUBIC)
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9, 1.1), shear=10, interpolation=InterpolationMode.BICUBIC,
                                    fill=0),
            transforms.Resize(self.output_size, InterpolationMode.NEAREST),
            transforms.RandomCrop((self.output_size, self.output_size), pad_if_needed=True, fill=0),
        ])
        self.pair_gt_dual_transform_nocrop = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9, 1.1), shear=10, interpolation=InterpolationMode.BICUBIC,
                                    fill=0),
            transforms.Resize(self.output_size, InterpolationMode.NEAREST)
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0.05),
            transforms.RandomGrayscale(0.05),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=(124, 116, 104)),
            transforms.RandomHorizontalFlip(),
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=0),
            transforms.RandomHorizontalFlip(),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=True)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img_pil = myutils.load_image_in_PIL(self.img_list[idx], 'RGB')
        mask_pil = myutils.load_image_in_PIL(self.mask_list[idx], 'P')

        frames = torch.zeros((self.clip_n, 3, self.output_size, self.output_size), dtype=torch.float)
        masks = torch.zeros((self.clip_n, self.max_obj_n, self.output_size, self.output_size), dtype=torch.float)

        sequence_seed = np.random.randint(2147483647)
        for i in range(self.clip_n):
            # img, mask = img_pil, mask_pil
            # if i > 0:
            #     img, mask = self.random_horizontal_flip(img, mask)
            #     img = self.color_jitter(img)
            #     img, mask = self.random_affine(img, mask)

            # img, mask = self.random_resize_crop(img, mask)

            # mask = np.array(mask, np.uint8)

            # if i == 0:
            #     mask, obj_list = self.to_onehot(mask)
            #     obj_n = len(obj_list) + 1
            #     img = self.to_tensor(img)
            # else:
            #     mask, _ = self.to_onehot(mask, obj_list)
            #     img = self.to_tensor(img)

            img, mask = img_pil, mask_pil
            dataset.reseed(sequence_seed)
            img = self.all_im_dual_transform(img)  # 仿射 翻转
            img = self.all_im_lone_transform(img)  # 颜色 灰度
            dataset.reseed(sequence_seed)
            mask = self.all_gt_dual_transform(mask)

            pairwise_seed = np.random.randint(2147483647)
            dataset.reseed(pairwise_seed)
            if not self.crop:
                img = self.pair_im_dual_transform_nocrop(img)  # 仿射 resize
                img = self.pair_im_lone_transform(img)  # 颜色
                dataset.reseed(pairwise_seed)
                mask = self.pair_gt_dual_transform_nocrop(mask)
            else:
                img = self.pair_im_dual_transform(img)  # 仿射 resize 裁剪
                img = self.pair_im_lone_transform(img)  # 颜色
                dataset.reseed(pairwise_seed)
                mask = self.pair_gt_dual_transform(mask)

            # Use TPS only some of the times
            # Not because TPS is bad -- just that it is too slow and I need to speed up data loading
            if np.random.rand() < 0.33:
                img, mask = dataset.random_tps_warp(img, mask, scale=0.02)
            mask = np.array(mask, np.uint8)

            if i == 0:
                mask, obj_list = self.to_onehot(mask)
                obj_n = len(obj_list) + 1
                img = self.to_tensor(img)
            else:
                mask, _ = self.to_onehot(mask, obj_list)
                img = self.to_tensor(img)

            frames[i] = img
            masks[i] = mask

        info = {
            'name': self.img_list[idx]
        }
        return frames, masks[:, :obj_n], obj_n, info


if __name__ == '__main__':
    pass
