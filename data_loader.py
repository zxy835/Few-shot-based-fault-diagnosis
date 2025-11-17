import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FaultDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_data(path1, path2):


    incipient = np.load(path1)
    severe = np.load(path2)

    assert incipient.shape[0] == severe.shape[0], "两个数据样本数不一致"

    labels = incipient[:, 0, -1].astype(np.int64) - 1
    incipient_data = incipient[:, :, :30]
    combined_data = np.concatenate((incipient_data, severe), axis=2)

    return combined_data, labels

def get_traindataloader(path1, path2, batch_size=32):
    data, labels = load_data(path1, path2)

    num = len(data)

    train_data = data[:int(num*0.8)]
    train_labels = labels[:int(num*0.8)]

    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    label_tensor = torch.tensor(train_labels, dtype=torch.long)

    dataset = FaultDataset(train_tensor, label_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def get_testdataloader(path1, path2, batch_size=32):
    data, labels = load_data(path1, path2)
    
    test_data = data[int(num*0.8):]
    test_labels = labels[int(num*0.8):]

    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    label_tensor = torch.tensor(test_labels, dtype=torch.long)

    dataset = FaultDataset(test_tensor, label_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def get_valdataloader(path3, batch_size=32):
    incipient = np.load(path3)

    features = incipient[:, :, :30]
    labels = incipient[:, 0, -1].astype(np.int64) - 1

    data_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = FaultDataset(data_tensor, label_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    return loader
