from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np
import glob
import os


class SocoFingerDataset(Dataset):
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
        # torch.from_numpy(np.array([y], dtype=np.float32))

    def __len__(self):
        return len(self.x)


class MyFingerDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, data_folder, transform=None, train=True):
        self.data_folder = data_folder
        self.train = train
        self.transform = transform
        files = glob.glob(self.data_folder)
        self.data = []
        self.labels = []
        for fp in files:
            x = Image.fromarray(np.load(fp) * 255, mode='L').resize((90, 90))
            label = os.path.basename(fp)
            l = label.rindex("_")
            label = label[:l]
            self.data.append(x)
            self.labels.append(label)
        self.labels = np.array(self.labels)

        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1, label1 = self.data[index], self.labels[index]
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(list(self.labels_set - {label1}))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        img2 = self.data[siamese_index]

        # img1 = Image.fromarray(img1 * 255, mode='L')
        # img2 = Image.fromarray(img2 * 255, mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.labels)
