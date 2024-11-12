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
    elif dataset_split == 'train':
        return train_samples
    elif dataset_split == 'val':
        return val_samples
    elif dataset_split == 'all':
        return train_samples, val_samples, test_samples
    elif dataset_split == 'all_combined':
        return curr_list

def get_organized_dataset_auxillary(base_dataset_path, dataset_name, dataset_split, llava_outdir):
    curr_list = get_dataset_list_auxillary(base_dataset_path, dataset_name, llava_outdir)
   
    if dataset_split == 'all_combined':
        return curr_list

    train_samples = [sample for sample in curr_list if sample['split'] == 'train']
    val_samples = [sample for sample in curr_list if sample['split'] == 'val']
    test_samples = [sample for sample in curr_list if sample['split'] == 'test']

    if dataset_split == 'test':
        return test_samples
    elif dataset_split == 'train':
        return train_samples
    elif dataset_split == 'val':
        return val_samples
    elif dataset_split == 'all':
        return train_samples, val_samples, test_samples

# Dataloader for CLIP
class CLIPDataloader(torch.utils.data.Dataset):
    def __init__(self, clip_transform, clip_tokenizer, learning_data: list):
        self.read_data = learning_data
        self.transform = clip_transform
        self.tokenizer = clip_tokenizer

    def __len__(self):
        return len(self.read_data)

    def __getitem__(self, idx):
        datapoint = self.read_data[idx]
        name = datapoint['name']
        path = datapoint['path']
        label = datapoint['label']
        place = datapoint['place']
        image = self.transform(Image.open(path))

        single_sample = {'name': name, 'image': image, 'label': label, 'place': place}

        if 'bground_text' in datapoint:
            bground_text = datapoint['bground_text']
            single_sample['bground_text'] = self.tokenizer(bground_text, 77, True)
        if 'object_text' in datapoint:
            object_text = datapoint['object_text']
            single_sample['object_text'] = self.tokenizer(object_text, 77, True)
        return single_sample