from ml.datasets import SocoFingerDataset
import torchvision.transforms as transforms
import numpy as np
from ml.models import EmbeddingNet, SiameseNet
from ml.losses import ContrastiveLoss
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from ml.trainer import fit
import torchvision.utils
from ml.utils import imshow
#from imgaug import augmenters as iaa


def vis_loader(vis_dataloader):
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)

    print(example_batch[2].numpy())
    imshow(torchvision.utils.make_grid(concatenated))


# def get_augmenters():
#     return iaa.Sequential([
#         iaa.GaussianBlur(sigma=(0, 0.5)),
#         iaa.Affine(
#             scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#             translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#             rotate=(-30, 30),
#             order=[0, 1],
#             cval=255
#         )
#     ], random_order=True)


if __name__ == '__main__':

    # test dataset generator
    x_real = np.load('dataset/x_real.npz')['data']
    y_real = np.load('dataset/y_real.npy')
    x_train = np.load('dataset/x_train.npy')
    x_val = np.load('dataset/x_val.npy')
    label_train = np.load('dataset/y_train.npy')
    label_val = np.load('dataset/y_val.npy')

    # x_data = np.concatenate([x_easy, x_medium, x_hard], axis=0)
    # label_data = np.concatenate([y_easy, y_medium, y_hard], axis=0)
    # x_train, x_val, label_train, label_val = train_test_split(
    #     x_data, label_data, test_size=0.1)

    print(x_train.shape, label_train.shape)
    print(x_val.shape, label_val.shape)

    label_real_dict = {}

    for i, y in enumerate(y_real):
        key = y.astype(str)
        key = ''.join(key).zfill(6)
        label_real_dict[key] = i

    train_dataset = SocoFingerDataset(x_train, label_train, x_real, label_real_dict,
                                      transform=transforms.Compose([
                                           transforms.ToPILImage(),
                                           transforms.RandomOrder([
                                               transforms.GaussianBlur(3,sigma=(0.1,0.5)),
                                                transforms.RandomAffine(
                                                        degrees =30,
                                                        translate=(0.1,0.1),
                                                        scale=(0.9, 1.1),     
                                                        fillcolor=255)
                                                ]),
                                           transforms.ToTensor()
                                       ]))

    test_dataset = SocoFingerDataset(x_val, label_val, x_real, label_real_dict,
                                     transform=transforms.ToTensor())

    cuda = torch.cuda.is_available()
    batch_size = 64
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  num_workers=6,
                                  batch_size=batch_size)

    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1, shuffle=True)

    # vis_loader(train_dataloader)

    margin = 1.
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = ContrastiveLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 15
    log_interval = 100
    
    fit(train_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    torch.save(model.state_dict(),
                   "./saved_models/model_siamese_3.pt")
    print("Model Saved Successfully")