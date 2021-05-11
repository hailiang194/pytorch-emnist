import csv
from torch.utils.data import Dataset
import torch
import numpy as np

class EMNISTDataset(Dataset):
    def __init__(self, csv_path):
        self.__label = []
        self.__dataset = []
        with open(csv_path, 'r') as csv_file:
            data = csv.reader(csv_file, delimiter=',')
            for row in data:
                self.__label.append(int(row[0]) - 1)
                img = torch.from_numpy((np.array([float(element) for element in row[1:]], dtype=np.float32)))
                self.__dataset.append(img)

    def __getitem__(self, index):
        return self.__dataset[index], self.__label[index]

    def __len__(self):
        return len(self.__label)

if __name__ == '__main__':
    dataset = EMNISTDataset('./test.csv')
    data, label = dataset[1]
    print(label)
