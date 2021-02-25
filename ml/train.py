import numpy as np
import random
import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.utils
from ml.utils import imshow, show_plot
import torchvision.transforms as transforms
from ml.losses import ContrastiveLoss
from ml.models import SiameseNetwork, SubtractNetwork
import torch.nn.functional as F
from ml.imgaug import augmenters as iaa
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from ml.datasets import FigurePrintDataset


EPOCHS = 15
BATCH_SIZE = 32
NET_TYPE = SubtractNetwork.__name__
# CHECKPOINT_PATH = "saved_models/model_state_SubtractNetwork_9.pt"
CHECKPOINT_PATH = None
TRAIN = True


def get_augmenters():
    return iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-30, 30),
            order=[0, 1],
            cval=255
        )
    ], random_order=True)


@torch.no_grad()
def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    predictions = []
    labels = []
    for i, data in enumerate(data_loader, 0):
        x0, x1, label = data
        score = None
        if NET_TYPE == SiameseNetwork.__name__:
            output1, output2 = model(x0.to(device), x1.to(device))
            score = F.pairwise_distance(output1, output2)
        else:
            output = model(x0.to(device), x1.to(device))
            score = torch.sigmoid(output)
        predictions.append(score.to(cpu_device).item())
        labels.append(label)

    predictions = np.array(predictions)
    labels = np.array(labels)
    return {'acc': accuracy_score(labels, predictions > 0.5), 'auc': roc_auc_score(labels, predictions)}


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

    train_dataset = FigurePrintDataset(x_train, label_train, x_real, label_real_dict,
                                       transform=transforms.Compose([
                                           get_augmenters().augment_image,
                                           transforms.ToTensor()
                                       ]))

    test_dataset = FigurePrintDataset(x_val, label_val, x_real, label_real_dict,
                                      transform=transforms.ToTensor())
    # Viewing the sample of images and to check whether its loading properly
    # vis_dataloader = DataLoader(train_dataset,
    #                             shuffle=True,
    #                             batch_size=8)
    # dataiter = iter(vis_dataloader)
    #
    # example_batch = next(dataiter)
    # concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    #
    # print(example_batch[2].numpy())
    # imshow(torchvision.utils.make_grid(concatenated))

    # train model

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  num_workers=4,
                                  batch_size=BATCH_SIZE)

    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if NET_TYPE == SiameseNetwork.__name__:
        net = SiameseNetwork()
        criterion = ContrastiveLoss()
    else:
        net = SubtractNetwork()
        criterion = nn.BCEWithLogitsLoss()
    net = net.to(device)
    # Declare Optimizer
    optimizer = torch.optim.Adam(
        net.parameters(), lr=1e-3, weight_decay=0.0005)

    loss = []
    counter = []
    iteration_number = 0
    loss_contrastive = None
    running_loss = 0.0
    start_epoch = 1

    if CHECKPOINT_PATH:
        print("Resume training")
        checkpoint = torch.load(CHECKPOINT_PATH)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']

    if TRAIN:
        for epoch in range(start_epoch, EPOCHS + 1):
            net.train()
            for i, data in enumerate(train_dataloader, 0):
                img0, img1, label = data
                img0, img1, label = img0.to(device), img1.to(
                    device), label.to(device)
                optimizer.zero_grad()
                if NET_TYPE == SiameseNetwork.__name__:
                    output1, output2 = net(img0, img1)
                    loss_contrastive = criterion(output1, output2, label)
                else:
                    output = net(img0, img1)
                    loss_contrastive = criterion(output, label)
                loss_contrastive.backward()
                optimizer.step()

                running_loss += loss_contrastive.item()
                if i % 300 == 299:  # every 1000 mini-batches...
                    print("Step {}, Loss {}\n".format(i, running_loss / 300))
                    running_loss = 0.0
            print("Epoch {}\n Current loss {}\n".format(
                epoch, loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss.append(loss_contrastive.item())
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, "saved_models/model_state_{}_{}.pt".format(NET_TYPE, epoch))

            val_results = evaluate(net, test_dataloader, device)
            print('Val Acc {}, AUC {}'.format(val_results['acc'], val_results['auc']))

        print(loss)
        show_plot(counter, loss)

        torch.save(net.state_dict(),
                   "./saved_models/model_.pt".format(NET_TYPE))
        print("Model Saved Successfully")

    # Validation

    val_results = evaluate(net, test_dataloader, device)
    print('Acc {}, AUC {}'.format(val_results['acc'], val_results['auc']))

    count = 0
    # for i, data in enumerate(test_dataloader, 0):
    #     x0, x1, label = data
    #     concat = torch.cat((x0, x1), 0)
    #     if NET_TYPE == SiameseNetwork.__name__:
    #         output1, output2 = net(x0.to(device), x1.to(device))
    #
    #         eucledian_distance = F.pairwise_distance(output1, output2)
    #     else:
    #         output = net(x0.to(device), x1.to(device))
    #         eucledian_distance = output
    #
    #     if label == torch.FloatTensor([[1]]):
    #         label = "Original Pair Of Signature"
    #     else:
    #         label = "Forged Pair Of Signature"
    #
    #     print("Predicted Eucledian Distance:-", eucledian_distance.item())
    #     print("Actual Label:-", label)
    #     imshow(torchvision.utils.make_grid(concat))
    #     count = count + 1
    #     if count == 5:
    #         break
