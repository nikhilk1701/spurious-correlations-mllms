# Code to: Read the dataset metadata file and get the list of paths and labels
# Date Created: 10/8/2024
# Last Modified By: Shika

from pathlib import Path
import pandas as pd
import os
import json

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

def get_dataset_list_auxillary(base_dataset_path: Path, dataset: str, llava_outdir: str):
    dataset = get_dataset_list(base_dataset_path, dataset)
    dataset_dict = {}

    for datapoint in dataset:
        dataset_dict[datapoint['name']] = datapoint

    bground_questions = llava_outdir + '/questions_bground.jsonl'
    object_questions = llava_outdir + '/questions_object.jsonl'
    bground_answers = llava_outdir + '/answers_bground.jsonl'
    object_answers = llava_outdir + '/answers_object.jsonl'

    with open(bground_answers, 'r') as afile, open(bground_questions, 'r') as qfile:
        for (qline, aline) in zip(qfile, afile):
            ques = json.loads(qline)
            ans = json.loads(aline)
            dataset_dict[ques['image']]['bground_text'] = ans['text']
    
    with open(object_answers, 'r') as afile, open(object_questions, 'r') as qfile:
        for (qline, aline) in zip(qfile, afile):
            ques = json.loads(qline)
            ans = json.loads(aline)
            dataset_dict[ques['image']]['object_text'] = ans['text']

    return [val for val in list(dataset_dict.values()) if 'object_text' in val and 'bground_text' in val]