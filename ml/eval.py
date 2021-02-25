import numpy as np
from ml.datasets import FigurePrintDataset
import torchvision.transforms as transforms
from ml.models import EmbeddingNet, SiameseNet
import torch
import torchvision.utils
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

def evaluate(model, data_loader):
    cpu_device = torch.device("cpu")
    predictions = []
    labels = []
    for i, data in enumerate(data_loader, 0):
        x, label = data
        x0, x1 = x
        concat = torch.cat((x0, x1), 0)
        output1, output2 = model(x0, x1)
        eucledian_distance = (output2 - output1).pow(2).sum(1)
        predictions.append(eucledian_distance.item())
        labels.append(label)

    predictions = np.array(predictions) < 0.1
    labels = np.array(labels)
    print(accuracy_score(labels,predictions))

def evaluate2(model, data_loader):
    cpu_device = torch.device("cpu")
    predictions = []
    labels = []
    for i, data in enumerate(data_loader, 0):
        x, label = data
        x0, x1 = x
        output = model(x0, x1)
        labels.append(label)
        predictions.append(output)

    predictions = np.array(predictions) > 0.5
    labels = np.array(labels)
    print(accuracy_score(labels,predictions))

if __name__ == '__main__':
    # test dataset generator
    x_real = np.load('dataset/x_real.npz')['data']
    y_real = np.load('dataset/y_real.npy')
    x_val = np.load('dataset/x_val.npy')
    label_val = np.load('dataset/y_val.npy')

    # x_data = np.concatenate([x_easy, x_medium, x_hard], axis=0)
    # label_data = np.concatenate([y_easy, y_medium, y_hard], axis=0)
    # x_train, x_val, label_train, label_val = train_test_split(
    #     x_data, label_data, test_size=0.1)

    print(x_val.shape, label_val.shape)

    label_real_dict = {}

    for i, y in enumerate(y_real):
        key = y.astype(str)
        key = ''.join(key).zfill(6)
        label_real_dict[key] = i
    
    test_dataset = FigurePrintDataset(x_val, label_val, x_real, label_real_dict,
                                      transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=1, shuffle=True)
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)
    weights = torch.load("./saved_models/model_siamese_3.pt",map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    #cpu_device = torch.device("cpu")
    #model = model.to(cpu_device)
    count = 0
    evaluate(model,test_dataloader)

    # for i, data in enumerate(test_dataloader, 0):
    #     x, label = data
    #     x0, x1 = x
    #     concat = torch.cat((x0, x1), 0)
    #     output1, output2 = model(x0, x1)
    #     eucledian_distance = (output2 - output1).pow(2).sum(1)
        
    #     if label == torch.FloatTensor([[1]]):
    #         label = "Original Pair Of Signature"
    #     else:
    #         label = "Forged Pair Of Signature"

    #     print("Predicted Eucledian Distance:-", eucledian_distance.item())
    #     print("Actual Label:-", label)
    #     imshow(torchvision.utils.make_grid(concat))
    #     count = count + 1
    #     if count == 5:
    #         break
