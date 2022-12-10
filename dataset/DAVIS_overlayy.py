import os
import random
import numpy as np
from glob import glob
# from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms as TF
from torchvision import transforms

from transforms import transforms as mytrans
import myutils
import dataset
from torchvision.transforms import InterpolationMode

MAX_TRAINING_SKIP = 25


class DAVIS17_Train(data.Dataset):

    def __init__(self, root, output_size, imset='2017/train.txt', clip_n=3, max_obj_n=11, increment=5,
                 max_skip=10, repeat_time=1):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.max_obj_n = max_obj_n
        self.increment = increment
        self.max_skip = max_skip
        self.repeat_time = repeat_time
        self.crop = True

        dataset_path = os.path.join(root, 'ImageSets', imset)
        self.dataset_list = list()
        with open(os.path.join(dataset_path), 'r') as lines:
            for line in lines:
                dataset_name = line.strip()
                if len(dataset_name) > 0:
                    self.dataset_list.append(dataset_name)

        # self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)
        # self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.02)
        # self.random_affine = mytrans.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=10)
        # self.random_resize_crop = mytrans.RandomResizedCrop(output_size, (0.8, 1), (0.95, 1.05))

        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])
        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=(124, 116, 104)),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])
        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((self.output_size, self.output_size), scale=(0.36, 1.00),
                                         interpolation=InterpolationMode.BICUBIC)
        ])
        self.all_im_dual_transform_nocrop = transforms.Compose([
            transforms.Resize(self.output_size, InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip()
        ])
        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((self.output_size, self.output_size), scale=(0.36, 1.00),
                                         interpolation=InterpolationMode.NEAREST)
        ])
        self.all_gt_dual_transform_nocrop = transforms.Compose([
            transforms.Resize(self.output_size, InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip()
        ])
        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=True)

    def __len__(self):
        return int(len(self.dataset_list) * self.repeat_time)

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def reload_max_skip(self):
        self.max_skip = 5

    def __getitem__(self, idx):

        idx = idx % len(self.dataset_list)
        video_name = self.dataset_list[idx]
        img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', '480p', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        last_frame = -1
        nframes = len(img_list)
        idx_list = list()

        for i in range(self.clip_n):
            if i == 0:
                last_frame = random.sample(range(0, nframes - self.clip_n + 1), 1)[0]

            else:
                last_frame = random.sample(
                        range(last_frame + 1, min(last_frame + self.max_skip + 1, nframes - self.clip_n + i + 1)),
                        1)[0]

            idx_list.append(last_frame)

        frames = torch.zeros((self.clip_n, 3, self.output_size, self.output_size), dtype=torch.float)
        masks = torch.zeros((self.clip_n, self.max_obj_n, self.output_size, self.output_size), dtype=torch.float)

        sequence_seed = np.random.randint(2147483647)
        for i, frame_idx in enumerate(idx_list):

            img_pil = myutils.load_image_in_PIL(img_list[frame_idx], 'RGB')
            mask_pil = myutils.load_image_in_PIL(mask_list[frame_idx], 'P')

            img, mask = img_pil, mask_pil
            dataset.reseed(sequence_seed)
            if not self.crop:
                img = self.all_im_dual_transform_nocrop(img)
                img = self.all_im_lone_transform(img)
                dataset.reseed(sequence_seed)
                mask = self.all_gt_dual_transform_nocrop(mask)
            else:
                img = self.all_im_dual_transform(img)
                img = self.all_im_lone_transform(img)
                dataset.reseed(sequence_seed)
                mask = self.all_gt_dual_transform(mask)

            pairwise_seed = np.random.randint(2147483647)
            dataset.reseed(pairwise_seed)
            img = self.pair_im_dual_transform(img)  # 仿射 resize 裁剪
            img = self.pair_im_lone_transform(img)  # 颜色
            dataset.reseed(pairwise_seed)
            mask = self.pair_gt_dual_transform(mask)


            mask = np.array(mask, np.uint8)

            if i == 0:
                mask, obj_list = self.to_onehot(mask)
                obj_n = len(obj_list) + 1
            else:
                mask, _ = self.to_onehot(mask, obj_list)

            frames[i] = self.to_tensor(img)
            masks[i] = mask
            # if i > 0:
            #     img = self.color_jitter(img)
            #     img, mask = self.random_affine(img, mask)

            # roi_cnt = 0
            # while roi_cnt < 10:
            #     img_roi, mask_roi = self.random_resize_crop(img, mask)

            #     mask_roi = np.array(mask_roi, np.uint8)

            #     if i == 0:
            #         mask_roi, obj_list = self.to_onehot(mask_roi)
            #         obj_n = len(obj_list) + 1
            #     else:
            #         mask_roi, _ = self.to_onehot(mask_roi, obj_list)

            #     if torch.any(mask_roi[0] == 0).item():
            #         break

            #     roi_cnt += 1

            # frames[i] = self.to_tensor(img_roi)
            # masks[i] = mask_roi

        info = {
            'name': video_name,
            'idx_list': idx_list
        }

        return frames, masks[:, :obj_n], obj_n, info


class DAVIS_Test(data.Dataset):

    def __init__(self, root, img_set='2017/val.txt', max_obj_n=11, single_obj=False):
        self.root = root
        self.single_obj = single_obj
        dataset_path = os.path.join(root, 'ImageSets', img_set)
        self.dataset_list = list()

        with open(os.path.join(dataset_path), 'r') as lines:
            for line in lines:
                dataset_name = line.strip()
                if len(dataset_name) > 0:
                    self.dataset_list.append(dataset_name)

        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=False)

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        video_name = self.dataset_list[idx]

        img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', '480p', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        first_mask = myutils.load_image_in_PIL(mask_list[0], 'P')
        first_mask_np = np.array(first_mask, np.uint8)

        if self.single_obj:
            first_mask_np[first_mask_np > 1] = 1

        h, w = first_mask_np.shape
        obj_n = first_mask_np.max() + 1
        video_len = len(img_list)

        frames = torch.zeros((video_len, 3, h, w), dtype=torch.float)
        masks = torch.zeros((video_len, obj_n, h, w), dtype=torch.float)

        

        for i in range(video_len):
            img = myutils.load_image_in_PIL(img_list[i], 'RGB')
            frames[i] = self.to_tensor(img)
            first_mask = myutils.load_image_in_PIL(mask_list[i], 'P')
            first_mask_np = np.array(first_mask, np.uint8)
            mask, _ = self.to_onehot(first_mask_np)
            masks[i] = mask[:obj_n]

        info = {
            'name': video_name,
            'num_frames': video_len,
        }

        return frames, masks, obj_n, info


if __name__ == '__main__':
    pass
