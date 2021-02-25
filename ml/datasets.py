from torch.utils.data import Dataset
import random
import torch
import numpy as np

class FigurePrintDataset(Dataset):
    def __init__(self, x, label, x_real, label_real_dict, transform=None):
        self.x = x
        self.label = label
        self.x_real = x_real
        self.label_real_dict = label_real_dict
        self.transform = transform

    def __getitem__(self, index):
        im1 = self.x[index]
        label = self.label[index]

        match_key = label.astype(str)
        match_key = ''.join(match_key).zfill(6)
        if random.random() > 0.5:
            im2 = self.x_real[self.label_real_dict[match_key]]
            y = 1
        else:
            while True:
                unmatched_key, unmatched_idx = random.choice(
                    list(self.label_real_dict.items()))
                if unmatched_key != match_key:
                    break
            im2 = self.x_real[unmatched_idx]
            y = 0

        # Apply image transformations
        if self.transform is not None:
            im1 = self.transform(im1)
            im2 = self.transform(im2)

        # Since torch work on C x H x W image
        return (im1, im2), y
        #torch.from_numpy(np.array([y], dtype=np.float32))

    def __len__(self):
        return len(self.x)
