# Code to: Read the dataset metadata file and get the list of paths and labels
# Date Created: 10/8/2024
# Last Modified By: Shika


from pathlib import Path
import pandas as pd
import os
import torch
import clip

def get_dataset_list(base_dataset_path: Path, dataset: str):
    datasets = []
    if dataset == 'waterbirds':
        curr_path = os.path.join(base_dataset_path, 'Waterbirds')
        label_file = os.path.join(curr_path, r'metadata.csv')
        pd_label_file = pd.read_csv(label_file)

        filenames = pd_label_file['img_filename'].tolist()
        ttv_split = pd_label_file['split'].tolist()
        gt_labels = pd_label_file['y'].tolist()
        places = pd_label_file['place'].tolist()

        label_place_match_count_train = 0
        label_place_match_count_val = 0
        label_place_match_count_test = 0

        for (_name, _label, _split, _place) in zip(filenames, gt_labels, ttv_split, places):
            dataset_entry = {}

            # get path for PIL
            dataset_entry['name'] = _name
            dataset_entry['path'] = os.path.join(curr_path, _name)
            
            # get label but not number label
            if "Eastern_Towhees" in _name or "Western_Meadowlarks" in _name or "Western_Wood_Pewees" in _name:
                dataset_entry['label'] = "landbird"
            else:
                if int(_label) == 0:
                    dataset_entry['label'] = "landbird"
                elif int(_label) == 1:
                    dataset_entry['label'] = "waterbird"
            
            # get test train split
            if int(_split) == 0:
                dataset_entry['split'] = "train"
            elif int(_split) == 1:
                dataset_entry['split'] = "val"
            elif int(_split) == 2:
                dataset_entry['split'] = "test"
            
            # get place
            if int(_place) == 0:
                dataset_entry['place'] = "land"
            elif int(_place) == 1:
                dataset_entry['place'] = "water"
            
            # get num of images in test where label and place matches
            if int(_label) == int(_place):
                if dataset_entry['split'] == 'test':
                    label_place_match_count_test += 1
                if dataset_entry['split'] == 'train':
                    label_place_match_count_train += 1
                if dataset_entry['split'] == 'val':
                    label_place_match_count_val += 1

            datasets.append(dataset_entry)

        print("number of datasets: ", len(datasets))

        # print num of train, test, val images
        print("Count of images in train where label (land/water) and place (land/water) match: ", label_place_match_count_train)
        print("Count of images in test where label (land/water) and place (land/water) match: ", label_place_match_count_test)
        print("Count of images in val where label (land/water) and place (land/water) match: ", label_place_match_count_val)
    return datasets


def get_text_data (base_dataset_path: Path, dataset: str):
    # Load the CSV file
    data = base_dataset_path / f"{dataset}"
    data = pd.read_csv(data)

    # Extract unique bird classes and their descriptions
    bird_classes = data['bird_class'].unique()
    descriptions = {row['bird_class']: row['description'] for _, row in data.iterrows()}

    # Create tokenized text inputs
    tokenized_text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in bird_classes])#.to(self.device)

    return tokenized_text_inputs, descriptions

if __name__ == "__main__":
    base_dataset_path = Path('D:/Downloads/Academics/Learning From Small Data/spurious-correlations-mllms')
    dataset           = 'dummy_class_data.csv'

    try:
        tokenized_text_inputs, descriptions = get_text_data(base_dataset_path, dataset)
        print("Tokenized Text Inputs:")
        print(tokenized_text_inputs)
        print("\nDescriptions:")
        for bird_class, description in descriptions.items():
            print(f"{bird_class}: {description}")
    except Exception as e:
        print(f"An error occurred: {e}")

