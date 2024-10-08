# Code to: Read the dataset metadata file and get the list of paths and labels
# Date Created: 10/8/2024
# Last Modified By: Shika


from pathlib import Path
import pandas as pd
import os

def get_dataset_list(base_dataset_path: Path, dataset: str):
    if dataset == 'waterbirds':
        names = []
        paths = []
        labels = []
        split = []
    
        curr_path = os.path.join(base_dataset_path, 'Waterbirds')
        label_file = os.path.join(curr_path, r'metadata.csv')
        pd_label_file = pd.read_csv(label_file)

        filenames = pd_label_file['img_filename'].tolist()
        ttv_split = pd_label_file['split'].tolist()
        gt_labels = pd_label_file['y'].tolist()
        places = pd_label_file['place'].tolist()

        for (name, label, split, place) in zip(filenames, gt_labels, ttv_split, places):

            label_place_match_count_test = 0

            # get path for PIL
            names.append(name)
            paths.append(os.path.join(curr_path, name))
            
            # get label but not number label
            if "Eastern_Towhees" in name or "Western_Meadowlarks" in name or "Western_Wood_Pewees" in name:
                labels.append("landbird")
            else:
                if int(label) == 0:
                    labels.append("landbird")
                elif int(label) == 1:
                    labels.append("waterbird")
            
            # get test train split
            if int(split) == 0:
                split.append("train")
            elif int(split) == 1:
                split.append("val")
            elif int(split) == 2:
                split.append("test")
            
            # get place
            if int(place) == 0:
                places.append("land")
            elif int(place) == 1:
                places.append("water")
            
            # get num of images in test where label and place matches
            if split == 'test' and int(label) == int(place):
                label_place_match_count_test += 1
        
        # print num of train, test, val images
        print("Count of images in test where label (land/water) and place (land/water) match: ", label_place_match_count_test)

    return {'names': names, 'paths': paths, 'labels': labels, 'split': split, 'places': places}