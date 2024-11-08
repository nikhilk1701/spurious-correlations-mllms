# Code to: Process the data for CLIP
# Date Created: 10/8/2024
# Last Modified By: Shika

from data_read_utils import *
import torch.utils.data
from PIL import Image

# Return relevant data
def get_organized_dataset(base_dataset_path, dataset_name, dataset_split):

    curr_list = get_dataset_list(base_dataset_path, dataset_name)
    train_samples = [sample for sample in curr_list if sample['split'] == 'train']
    val_samples = [sample for sample in curr_list if sample['split'] == 'val']
    test_samples = [sample for sample in curr_list if sample['split'] == 'test']

    if dataset_split == 'test':
        return test_samples
    elif dataset_split == 'all':
        return train_samples, val_samples, test_samples
    elif dataset_split == 'all_combined':
        return curr_list

# Dataloader for CLIP
class CLIPDataloader(torch.utils.data.Dataset):
    def __init__(self, clip_transform, learning_data: dict):
        self.read_data = learning_data
        self.transform = clip_transform

    def __len__(self):
        return len(self.read_data)

    def __getitem__(self, idx):
        name            = self.read_data[idx]['name']
        path            = self.read_data[idx]['path']
        label           = self.read_data[idx]['label']
        tokenized_label = self.read_data[idx]['tokenized_label']
        tokenized_desc  = self.read_data[idx]['tokenized_desc']
        image = self.transform(Image.open(path))

        single_sample = {'name': name, 'image': image, 'label': label,
                         'tokenized_label': tokenized_label, 'tokenized_desc': tokenized_desc}

        return single_sample