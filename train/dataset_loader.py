import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class MaskGenerator:
    def __init__(self, n_wave):
        self.n_wave = n_wave
    def __call__(self, mask_type):
        if mask_type == 0:
            mask_type = np.random.randint(1, 6)
        mask = torch.zeros(self.n_wave, 6)
        if mask_type == 1:
            w_idx = np.random.randint(0, self.n_wave)
            mask[w_idx, :] = 1
        elif mask_type == 2:
            mask[:, 3:] = 1
        elif mask_type == 3:
            mask[:, 0] = 1
            mask[:, 3] = 1
        elif mask_type == 4:
            mask[:, 1] = 1
            mask[:, 4] = 1
        elif mask_type == 5:
            mask[:, [1, 2, 4, 5]] = 1
        return mask

class JonesMatrixDataset(Dataset):
    def __init__(self, jm_array, n_wave, mask_type=0):
        self.jm_array = jm_array
        self.n_wave = n_wave
        self.mask_gen = MaskGenerator(n_wave)
        self.mask_type = mask_type
    def __len__(self):
        return self.jm_array.shape[0]
    def __getitem__(self, idx):
        img = torch.from_numpy(self.jm_array[idx]).view(self.n_wave, 6).unsqueeze(0).float()
        mask = self.mask_gen(self.mask_type)
        return img, mask

def load_pretrain_dataloaders(dataset_dir, batch_size=256, val_ratio=0.1, mask_type=0, num_workers=4):
    jm_path = os.path.join(dataset_dir, "JM_dimer.csv")
    wave_path = os.path.join(dataset_dir, "wavelengths_nm.csv")
    
    jm_df = pd.read_csv(jm_path)
    n_wave = len(pd.read_csv(wave_path))
    jm_array = jm_df.to_numpy()
    
    full_dataset = JonesMatrixDataset(jm_array, n_wave, mask_type=mask_type)
    n_val = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])
    
    # RTX 5070 Ti এর জন্য pin_memory=True করা হয়েছে
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader