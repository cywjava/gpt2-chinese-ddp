import json
import torch
import random
from torch.utils.data import Dataset

"""
    自定义dataset
    author:chen.yiwan 
    date:2023-06-05
"""


class GPT2DataSet(Dataset):
    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item_data = self.data[index]
        return item_data

    @staticmethod
    def load_json(file_list):
        result_data = []
        for jf in file_list:
            with open(jf, 'r', encoding="utf-8") as file:
                try:
                    result_data = json.load(file)
                except Exception as e:
                    print("load json file,occurring format error, file:", jf)
                    exit(0)
            random.shuffle(result_data)
        return list(result_data)

    def __len__(self):
        return len(self.data)
