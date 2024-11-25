# Code to: Evaluate CLIP on waterbirds
# Date Created: 10/8/2024
# Last Modified By: Shika

import traceback
import datetime
import time
from tqdm import tqdm
import torch.utils.data
from dataloader import *
import torch
import clip
import pandas as pd
import numpy as np
from networks import *

def eval_mode(model):
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model

def load_clip_model(model_type, text_classes, device):
    model, preprocess = clip.load(model_type, device=device)
    model = eval_mode(model)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in text_classes]).to(device)
    return model, preprocess, text_inputs, device

def load_pretrained_model(clip_model, load_path, device):
    # copy paste from the corresponding train file to load the model
    model_dict = torch.load(load_path)
    clip_model.load_state_dict(model_dict['model']['state_dict'])
    model = clip_model.to(device)
    return model

def clip_inference(loaded_model, text_inputs, test_loader, device, binary=False):
    with torch.no_grad():
        img_name = []
        gt_labels = []
        scores = []

        for sampled_batch in tqdm(test_loader):
            img = sampled_batch['image']
            name = sampled_batch['name']
            label = sampled_batch['label']

            img = img.to(device)

            logits_per_image, logits_per_text = loaded_model(img, text_inputs)
            if binary:
                selected_logits = logits_per_image[:, 0].unsqueeze(1)  # Now shape: [32, 1] as in training
                preds = torch.sigmoid(selected_logits).detach().cpu().numpy()
                probs = (preds > 0.5).astype(int)
            else:
                probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

            img_name.extend(name)
            gt_labels.extend(label)
            scores.extend(probs)

    return img_name, gt_labels, scores

def save_to_csv(model_type, dataset, save_path, img_name, gt_labels, scores):
    df = pd.DataFrame({'img_name': img_name, 'gt_labels': gt_labels, 'scores': [str(score) for score in scores]})
    df.to_csv(f"{save_path}/CLIP_{model_type}_{dataset}_inference.csv", index= False)
    return

def calculate_class_pred_accuracy(text_class_order, gt_labels, scores, binary=False):

    if binary:
        class_predictions = [text_class_order[int(score)] for score in scores]
    else:
        class_predictions = [text_class_order[np.argmax(score)] for score in scores]
    
    # Calculate overall accuracy    
    correct_predictions = sum(pred == gt for pred, gt in zip(class_predictions, gt_labels))
    accuracy = correct_predictions / len(gt_labels)

    # calculate accuracy per class
    class_accuracy = {}
    for class_name in text_class_order:
        class_indices = [i for i, gt in enumerate(gt_labels) if gt == class_name]
        if len(class_indices) == 0:
            continue
        class_gt_labels = [gt_labels[i] for i in class_indices]
        tmp = [class_predictions[i] for i in class_indices]
        correct_class_predictions = sum(pred == gt for pred, gt in zip(tmp, class_gt_labels))
        class_accuracy[class_name] = correct_class_predictions / len(class_indices)

    return accuracy, class_accuracy

def calculate_4class_pred_accuracy(dataset, scores, class_order, binary=False):
    total_counts = {
        "waterbird": { "land": 0, "water": 0, },
        "landbird": { "land": 0, "water": 0, },
    }
    correct_counts = {
        "waterbird": { "land": 0, "water": 0, },
        "landbird": { "land": 0, "water": 0, },
    }
    for datapoint, score in zip(dataset, scores):
        if binary:
            guess = class_order[int(score)]
        else:
            guess = class_order[0] if score[0] > score[1] else class_order[1]

        gt_label = datapoint['label']
        gt_place = datapoint['place']

        if guess == gt_label:
            correct_counts[guess][gt_place] += 1
        
        total_counts[gt_label][gt_place] += 1
    
    accuracy = { }

    for gt_class in ["waterbird", "landbird"]:
        accuracy[gt_class] = {}
        for place in ["water", "land"]:
            if total_counts[gt_class][gt_place] != 0:
                accuracy[gt_class][place] = (correct_counts[gt_class][place] * 1.0) / total_counts[gt_class][gt_place]
    return accuracy

def main_eval(mode, network, load_path, layer_type_image, layer_type_text, binary_mode):
    text = ["waterbird", "landbird"]
    dataset = "waterbirds"
    scratch_dir = os.getenv("SCRATCH")
    img_dir = scratch_dir + '/datasets'
    model_type = "ViT-B/32"
    save_path = scratch_dir + '/results'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Added the functionality to evaluate not just zeroshot clip model but also a finetuned clip model
    clip_model, preprocess, text_inputs, device = load_clip_model(model_type, text, device= device)
    if mode == 'pretrained':
        if network == 'clip':
            model = clip_model
        elif network == 'modified_clip':
            model = CLIPCombinedModified(clip_model, layer_type_image= layer_type_image, layer_type_text= layer_type_text) 
        model = load_pretrained_model(model, load_path, device)
    else:
        model = clip_model

    read_dataset = get_organized_dataset(base_dataset_path=Path(img_dir), dataset_name=dataset, dataset_split='test')
    loaded_dataset = CLIPDataloader(clip_transform= preprocess, learning_data= read_dataset, clip_tokenizer=clip.tokenize)
    loader = torch.utils.data.DataLoader(loaded_dataset, batch_size=32, shuffle=False)
    
    img_name, gt_labels, scores = clip_inference(model, text_inputs, loader, device, binary_mode)
    save_to_csv(model_type.split("/")[0], dataset, save_path, img_name, gt_labels, scores)
    accuracy, class_accuracy = calculate_class_pred_accuracy(text, gt_labels, scores, binary_mode)
    subclass_accuracy = calculate_4class_pred_accuracy(read_dataset, scores, text, binary_mode)

    print(f"Overall Accuracy: {accuracy}")
    for class_name, accuracy in class_accuracy.items():
        print(f"{class_name} Accuracy: {accuracy}")

    for c_name, c_acc in subclass_accuracy.items():
        for subc_name, subc_acc in c_acc.items():
            print(f"{c_name} on {subc_name} background accuracy: {subc_acc}")

    return

def main():
    mode = 'clip'
    network = 'dummy'
    load_path = 'dummy'
    layer_type_image = 'dummy'
    layer_type_text = 'dummy'
    binary_mode = False
    main_eval(mode, network, load_path, layer_type_image, layer_type_text, binary_mode)

if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))