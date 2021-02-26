from torch.utils.data import Dataset
import random
import numpy as np


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

    def __init__(self, train = True):
        self.mnist_dataset = mnist_dataset

        self.train = train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)