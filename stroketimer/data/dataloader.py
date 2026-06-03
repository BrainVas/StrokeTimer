# -*- coding: utf-8 -*-
"""
Dataloader with 'ncct_dataset' 3D support.
Adds: balance_eval flag to ensure strict per-class down-sampling for val/test.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

from stroketimer.data.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
# 3D NCCT dataset (center-stratified version)
from stroketimer.data.ncct_center_dataset import NCCTDataset

# Image statistics kept for other datasets
RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}


def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key != 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


class LT_Dataset(Dataset):
    """2D long-tail list dataset (kept for compatibility)."""
    def __init__(self, root, txt, transform=None, template=None, top_k=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        if top_k:
            if 'train' in txt:
                max_len = max(self.labels) + 1
                dist = [[i, 0] for i in range(max_len)]
                for i in self.labels:
                    dist[i][-1] += 1
                dist.sort(key=lambda x: x[1], reverse=True)
                torch.save(dist, template + '_top_{}_mapping'.format(top_k))
            else:
                dist = torch.load(template + '_top_{}_mapping'.format(top_k))
            selected_labels = {item[0]: i for i, item in enumerate(dist[:top_k])}
            new_img_path, new_labels = [], []
            for path, label in zip(self.img_path, self.labels):
                if label in selected_labels:
                    new_img_path.append(path)
                    new_labels.append(selected_labels[label])
            self.img_path = new_img_path
            self.labels = new_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, index

    def get_num_classes(self):
        return max(self.labels) + 1 if len(self.labels) > 0 else 0

    def get_annotations(self):
        return [{'category_id': int(l)} for l in self.labels]

    def get_cls_num_list(self):
        from collections import Counter
        c = Counter(self.labels)
        return [c.get(i, 0) for i in range(self.get_num_classes())]


def load_data(
    data_root,
    dataset,
    phase,
    batch_size,
    top_k_class=None,
    sampler_dic=None,
    num_workers=4,
    shuffle=True,
    cifar_imb_ratio=None,
    # ---- ncct-specific passthrough ----
    ncct_csv_path: str = None,
    ncct_split_ratios=(0.7, 0.15, 0.15),
    ncct_seed: int = 42,
    ncct_transform=None,          # external 3D transform (TorchIO/own), optional
    ncct_fix_depth: bool = True,  # enforce D=48 by crop/pad
    ncct_zscore: bool = True,     # per-volume z-score in dataset
    ncct_balance_eval: bool = True,  # NEW: strictly balance val/test via down-sampling
):
    """
    Unified loader.

    For 'ncct_dataset' (3D):
      - No 2D torchvision transforms are constructed here.
      - Pass your 3D transform via 'ncct_transform' if needed.
      - Dataset always returns (1,48,256,256) float32 tensors.
      - 'ncct_balance_eval' controls strict per-class down-sampling for val/test.
    """
    txt_split = phase
    template = './data/%s/%s' % (dataset, dataset)
    print(f'Loading dataset={dataset} phase={phase} from {data_root}')

    if dataset == 'CIFAR10_LT':
        print('====> CIFAR10 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR10(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)

    elif dataset == 'CIFAR100_LT':
        print('====> CIFAR100 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR100(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)

    elif dataset == 'ncct_dataset':
        # 3D NCCT dataset; all split logic is inside NCCTDataset
        set_ = NCCTDataset(
            root=data_root,
            phase=phase,                    # 'train' | 'val' | 'test'
            transform=ncct_transform,
            split_ratios=ncct_split_ratios,
            seed=ncct_seed,
            verbose=True,
            csv_path=ncct_csv_path,
            fix_depth=ncct_fix_depth,
            zscore=ncct_zscore,
            balance_eval=ncct_balance_eval,
        )

    else:
        # Default LT text-list datasets (ImageNet_LT, Places_LT, etc.)
        key = 'default'
        rgb_mean = RGB_statistics[key]['mean']
        rgb_std = RGB_statistics[key]['std']
        txt = './data/%s/%s_%s.txt' % (dataset, dataset, txt_split)
        print('Loading image-list from %s' % (txt))
        if phase not in ['train', 'val']:
            transform = get_data_transform('test', rgb_mean, rgb_std, key)
        else:
            transform = get_data_transform(phase, rgb_mean, rgb_std, key)
        print('Use data transformation:', transform)
        set_ = LT_Dataset(data_root, txt, transform, template=template, top_k=top_k_class)

    print(len(set_))

    if sampler_dic and phase == 'train':
        print('=====> Using sampler: ', sampler_dic['sampler'])
        print('=====> Sampler parameters: ', sampler_dic['params'])
        return DataLoader(
            dataset=set_,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        # For val/test we generally set shuffle=False for reproducibility
        eff_shuffle = shuffle if phase == 'train' else False
        print('=====> No sampler.')
        print('=====> Shuffle is %s.' % (eff_shuffle))
        return DataLoader(
            dataset=set_,
            batch_size=batch_size,
            shuffle=eff_shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
