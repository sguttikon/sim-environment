#!/usr/bin/env python3

import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def calc_odometry(old_pose, new_pose):
    x1, y1, th1 = old_pose
    x2, y2, th2 = new_pose

    abs_x = (x2 - x1)
    abs_y = (y2 - y1)
    sin = np.sin(th1)
    cos = np.cos(th1)

    odom_th = wrap_angle(th2 - th1)
    odom_x = cos * abs_x + sin * abs_y
    odom_y = sin * abs_x - cos * abs_y

    return np.array([odom_x, odom_y, odom_th])

class TransitionModelDataSet(Dataset):

    def __init__(self, pkl_file):

        with open(pkl_file, 'rb') as file:
            self.pkl_data = pickle.load(file)

    def __len__(self):
        return len(self.pkl_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx == 0:
            prev_sample = self.pkl_data[idx]
        else:
            prev_sample = self.pkl_data[idx-1]
        curr_sample = self.pkl_data[idx]

        sample = {}
        sample['old_pose'] = prev_sample['pose']
        sample['new_pose'] = curr_sample['pose']
        sample['vel_cmd'] = curr_sample['vel_cmd']
        sample['delta_t'] = curr_sample['delta_t']

        return sample

if __name__ == '__main__':
    file_idx = 0
    pkl_file_name = '../supervised/igibson_data/rnd_pose_obs_data/data_{:04d}.pkl'.format(file_idx)
    tm_dataset = TransitionModelDataSet(pkl_file=pkl_file_name)

    tm_data_loader = DataLoader(tm_dataset, batch_size=1, shuffle=True, num_workers=0)

    for _, batch_samples in enumerate(tm_data_loader):
        print(batch_samples)
        break
