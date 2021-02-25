import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    y_real = np.load('dataset/y_real.npy')
    x_easy = np.load('dataset/x_easy.npz')['data']
    y_easy = np.load('dataset/y_easy.npy')
    x_medium = np.load('dataset/x_medium.npz')['data']
    y_medium = np.load('dataset/y_medium.npy')
    x_hard = np.load('dataset/x_hard.npz')['data']
    y_hard = np.load('dataset/y_hard.npy')

    x_data = np.concatenate([x_easy, x_medium, x_hard], axis=0)
    label_data = np.concatenate([y_easy, y_medium, y_hard], axis=0)
    x_train, x_val, label_train, label_val = train_test_split(
        x_data, label_data, test_size=0.1)

    print(x_data.shape, label_data.shape)
    print(x_train.shape, label_train.shape)
    print(x_val.shape, label_val.shape)

    np.save('dataset/x_train.npy', x_train)
    np.save('dataset/y_train.npy', label_train)
    np.save('dataset/x_val.npy', x_val)
    np.save('dataset/y_val.npy', label_val)
    print('done')