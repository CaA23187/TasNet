import h5py
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, hdf5path, type):
        assert type in ['train','test']
        with h5py.File(hdf5path, 'r') as hf:
            self.src_1 = hf[type]['src_1'][:]
            self.src_2 = hf[type]['src_2'][:]
            self.mix = hf[type]['mix'][:]
        self.len = len(self.src_1)

    def __getitem__(self, i):
        index = i % self.len
        src1 = self.src_1[index]
        src2 = self.src_2[index]
        mix = self.mix[index]
        src1, src2, mix = self.data_preproccess((src1, src2, mix))

        return src1, src2, mix

    def __len__(self):
        return self.len

    def data_preproccess(self, data):
        """
        数据预处理
        :param data:
        :return:
        """
        s1, s2, mix = data

        s1 = torch.from_numpy(s1)
        s2 = torch.from_numpy(s2)
        mix = torch.from_numpy(mix)

        return s1, s2, mix
