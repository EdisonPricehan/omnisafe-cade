import os

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from collections import deque
import cv2


class MixedSequenceReader:
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        csv_path_dict: Dict[str, str],
        horizon: int = 8,
    ):
        assert os.path.exists(images_dir)
        assert os.path.exists(masks_dir)
        for k, v in csv_path_dict.items():
            assert os.path.exists(v)
        assert horizon > 0

        # Define constants
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.csv_path_dict = csv_path_dict
        self.horizon = horizon

        # Define variables
        self.image_queue = deque(maxlen=horizon)
        self.mask_queue = deque(maxlen=horizon)
        self.obs_queue = deque(maxlen=horizon)
        self.act_queue = deque(maxlen=horizon)
        self.cursor = 0
        self.started = False

        # Load images and masks paths
        img_entries = list(os.scandir(images_dir))
        img_entries.sort(key=lambda item: item.name)
        self.img_paths = [e.path for e in img_entries if e.is_file()]

        mask_entries = list(os.scandir(masks_dir))
        mask_entries.sort(key=lambda item: item.name)
        self.mask_paths = [e.path for e in mask_entries if e.is_file()]

        self.n = len(self.img_paths)
        if self.n != len(self.mask_paths):
            raise ValueError(f'Images number {self.n} and masks number {len(self.mask_paths)} do not match!')

        if horizon > self.n:
            raise ValueError(f'Data len should be greater than horizon!')

        # Load data from csv
        self.method_df_dict = {}
        for k, v in self.csv_path_dict.items():
            self.method_df_dict[k] = pd.read_csv(v)

        for k, df in self.method_df_dict.items():
            if self.n != len(df):
                raise ValueError('Images-masks number and dataframe number do not match!')

    def __iter__(self):
        self.cursor = 0
        self.image_queue.clear()
        self.mask_queue.clear()
        self.obs_queue.clear()
        self.act_queue.clear()
        self.started = False
        return self

    def __next__(self):
        if self.cursor >= self.n:
            raise StopIteration

        # Get csv data
        method_obs_act_dict = self.get_obs_act()

        # Get image and mask data
        if self.started:
            img = cv2.imread(self.img_paths[self.cursor])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_paths[self.cursor], cv2.IMREAD_GRAYSCALE)
            self.image_queue.append(img)
            self.mask_queue.append(mask)

            self.cursor += 1
        else:
            for i in range(self.horizon):
                img = cv2.imread(self.img_paths[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
                self.image_queue.append(img)
                self.mask_queue.append(mask)

            self.started = True
            self.cursor += self.horizon

        return list(self.image_queue), list(self.mask_queue), method_obs_act_dict

    def get_obs_act(self):
        method_data_dict = {}

        if self.started:
            start: int = self.cursor - self.horizon + 1
            end: int = self.cursor + 1
        else:
            start: int = 0
            end: int = self.horizon

        for k, df in self.method_df_dict.items():
            df_slice = df.iloc[start: end]

            obs_list = df_slice['observation'].apply(lambda s: np.fromstring(s.strip("[]"), sep=" ", dtype=float)).to_list()

            act_list = df_slice['action'].apply(lambda s: np.fromstring(s.strip("[]"), sep=" ", dtype=float)).to_list()

            method_data_dict[k] = {
                'observation': obs_list,
                'action': act_list,
            }

        return method_data_dict


if __name__ == '__main__':
    # Define paths
    inference_dir = './evaluations/riverine/video_inference'
    images_dir: str = os.path.join(inference_dir, 'images_wildcat')
    masks_dir: str = os.path.join(inference_dir, 'masks_wildcat')
    eval_stats_dict = {
        'mgae': os.path.join(inference_dir, 'mgae_wildcat.csv'),
        'lagrangian': os.path.join(inference_dir, 'lagrangian_wildcat.csv'),
        'safety_layer': os.path.join(inference_dir, 'safety_layer_wildcat.csv'),
    }

    # Init sequence reader
    msr = MixedSequenceReader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        csv_path_dict=eval_stats_dict,
        horizon=10,
    )

    for idx, data in enumerate(msr):
        print(f'{idx=}')
        # print(f'{len(data)=}')
        # print(f'{len(data[0])=}')
        # print(f'{len(data[1])=}')
        # print(f'{len(data[2])=}')
        # print(f'{data[2].keys()=}')
        print(f'{data[2]["mgae"]["observation"]}')
        # print(f'{data[2]=}')

        if idx == 14:
            exit(0)

        print('-' * 50)

