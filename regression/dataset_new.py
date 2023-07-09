import torch
from torch_geometric.data import InMemoryDataset, Dataset
import os.path as osp
import os

class GNNDataset(Dataset):

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        if train:
            self.data_div = 'train'
        else:
            self.data_div = 'test'
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_test.csv', 'train_prot5.pth', 'test_prot5.pth']

    @property
    def processed_file_names(self):
        processed_dir = osp.join(self.root, 'processed', self.data_div)
        return [osp.join(processed_dir, filename) for filename in os.listdir(processed_dir)]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process(self):
        pass

    def len(self):
        processed_dir = osp.join(self.root, 'processed', self.data_div)
        return len(os.listdir(processed_dir))
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'{self.data_div}/processed_data_{self.data_div}_{idx}.pt'))
        return data


if __name__ == "__main__":
    dataset = GNNDataset('data/davis')


