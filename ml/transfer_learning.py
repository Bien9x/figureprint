from torch.utils.data import DataLoader
from ml.datasets import MyFingerDataset
import torchvision.transforms as transforms

from ml.losses import ContrastiveLoss
from ml.models import EmbeddingNet, SiameseNet
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from ml.trainer import fit


def train(train_folder="data/process_images/*.npy", test_folder="data/test_images/*.npy",
          save_path="models/model_siamese_transfer_learning.pt"):
    train_dataset = MyFingerDataset(train_folder, transform=transforms.Compose([
        transforms.RandomOrder([
            transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
            transforms.RandomAffine(
                degrees=30,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                fillcolor=255)
        ]),
        transforms.ToTensor()
    ]))
    train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=5, shuffle=True)

    test_dataset = MyFingerDataset(test_folder, transform=transforms.ToTensor()
                                   )
    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=1, shuffle=False)

    # training
    margin = 1.
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    loss_fn = ContrastiveLoss(margin)
    lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 2
    log_interval = 1
    print('Start training')
    fit(train_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    torch.save(model.state_dict(),
               save_path)
    print("Model Saved Successfully")
