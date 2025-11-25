# loader.py
import numpy as np
import os
from torch.utils.data import Dataset
from skimage import io

class BeamCKM(Dataset):
    """DataLoader, once one beam"""
    def __init__(self, maps_inds=np.zeros(1), phase="train", dir_dataset="./BeamCKMSeer/", numTx=100, simulation="data", p=0.01, transform=None, para = True):
        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 100, 1, dtype=np.int16)
        else:
            self.maps_inds = maps_inds
        if phase == "train":
            self.ind1, self.ind2 = 0, 79
        elif phase == "val":
            self.ind1, self.ind2 = 80, 89
        elif phase == "test":
            self.ind1, self.ind2 = 90, 99
        else:
            raise ValueError("phase must be 'train', 'val' or 'test'")

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.simulation = simulation
        self.p = p
        self.num_beams = 8
        self.transform = transform
        
        self.dir_buildings = os.path.join(dir_dataset, "png/buildings_complete/")
        self.dir_Tx = os.path.join(dir_dataset, "png/antennas/")
        self.dir_gain = os.path.join(dir_dataset, f"{simulation}/")
        self.noise_std = 0 # Add Gaussian noise
        self.mask_size = 0 # Add Mask
        self.para = para

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx * self.num_beams

    def __getitem__(self, idx):
        idx_beam = idx % self.num_beams
        idx_orig_for_seed = idx // self.num_beams
        idxr = idx_orig_for_seed // self.numTx
        idxc = idx_orig_for_seed % self.numTx
        map_id = self.maps_inds[idxr + self.ind1]

        input_buildings = io.imread(os.path.join(self.dir_buildings, f"{map_id}.png")) / 255.0
        if self.noise_std > 0 and self.para == True:
            noise = np.random.normal(0, self.noise_std, input_buildings.shape)
            input_buildings = np.clip(input_buildings + noise, 0, 1)     
        
        if self.mask_size > 0 and self.para == True:
            h, w = input_buildings.shape
            x = np.random.randint(0, w - self.mask_size + 1)
            y = np.random.randint(0, h - self.mask_size + 1)
            mask = np.zeros_like(input_buildings)
            mask[y:y+self.mask_size, x:x+self.mask_size] = -1
            input_buildings = np.where(mask == -1, -1, input_buildings)
            
        input_Tx = io.imread(os.path.join(self.dir_Tx, f"{map_id}_{idxc}.png")) / 255.0
        
        gain_path_prev1 = os.path.join(self.dir_gain, f"{map_id}_{idxc}/{(idx_beam - 1) % self.num_beams}.png")
        image_gain_prev1 = io.imread(gain_path_prev1) / 255.0
        gain_path_current = os.path.join(self.dir_gain, f"{map_id}_{idxc}/{idx_beam}.png")
        image_gain_current = io.imread(gain_path_current) / 255.0
        gain_path_next1 = os.path.join(self.dir_gain, f"{map_id}_{idxc}/{(idx_beam + 1) % self.num_beams}.png")
        image_gain_next1 = io.imread(gain_path_next1) / 255.0
        np.random.seed(idx_orig_for_seed * self.num_beams + idx_beam)

        sample_mask_prev1 = np.random.choice([0, 1], size=(256, 256), p=[1-self.p, self.p])
        sample_mask_current = np.random.choice([0, 1], size=(256, 256), p=[1-self.p, self.p])
        sample_mask_next1 = np.random.choice([0, 1], size=(256, 256), p=[1-self.p, self.p])
        
        sampled_gain_prev1 = np.where(sample_mask_prev1, image_gain_prev1, -1.0)
        sampled_gain_current = np.where(sample_mask_current, image_gain_current, -1.0)
        sampled_gain_next1 = np.where(sample_mask_next1, image_gain_next1, -1.0)

        extended_input = np.stack([
            input_Tx,
            input_buildings,
            sampled_gain_prev1,
            sampled_gain_current,
            sampled_gain_next1,
        ], axis=0).astype(np.float32)

        combined_data = np.concatenate([extended_input, image_gain_current[np.newaxis, ...]], axis=0)
        if self.transform:
            combined_data = self.transform(combined_data)
        extended_input = combined_data[:len(extended_input)]
        image_gain_current = combined_data[-1:]
        return extended_input, image_gain_current
    