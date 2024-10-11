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

def eval_mode(model):
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model

def load_model(model_type, text_classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_type, device=device)
    model = eval_mode(model)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in text_classes]).to(device)
    return model, preprocess, text_inputs, device

def clip_inference(loaded_model, text_inputs, test_loader, device):
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
            probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

            img_name.append(name)
            gt_labels.append(label)
            scores.append(probs)

    return img_name, gt_labels, scores

def save_to_csv(model_type, dataset, save_path, img_name, gt_labels, scores):
    df = pd.DataFrame({'img_name': img_name, 'gt_labels': gt_labels, 'scores': [str(score) for score in scores]})
    df.to_csv(f"{save_path}/CLIP_{model_type}_{dataset}_inference.csv", index= False)
    return

def calculate_class_pred_accuracy_from_csv(text_class_order, gt_labels, scores):
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


def main():
    text = ["waterbird", "landbird"]
    dataset = "waterbirds"
    scratch_dir = os.getenv("SCRATCH")
    img_dir = scratch_dir + '/datasets'
    model_type = "ViT-B/32"
    save_path = scratch_dir + '/results'

    model, preprocess, text_inputs, device = load_model(model_type, text)
    read_dataset = get_organized_dataset(base_dataset_path=Path(img_dir), dataset_name=dataset, dataset_split='test')
    loaded_dataset = CLIPDataloader(clip_transform= preprocess, learning_data= read_dataset)
    loader = torch.utils.data.DataLoader(loaded_dataset, batch_size=32, shuffle=False)
    
    img_name, gt_labels, scores = clip_inference(model, text_inputs, loader, device)
    img_arr = []
    gt_label_arr = []
    score_arr = []
    for img_batch in img_name:
        img_arr += img_batch
    for gt_label_batch in gt_labels:
        gt_label_arr += gt_label_batch
    for score_batch in scores:
        if len(score_arr) == 0:
            score_arr = score_batch
        else:
            score_arr = np.concatenate([score_arr, score_batch])
    save_to_csv(model_type, dataset, save_path, img_arr, gt_label_arr, score_arr)

    csv_save_path = f"{save_path}/CLIP_{model_type}_{dataset}_inference.csv"
    df = pd.read_csv(csv_save_path)
    gt_labels = df['gt_labels'].tolist()
    # gt_labels = [label for label in gt_labels]
    scores = df['scores'].tolist()
    accuracy, class_accuracy = calculate_class_pred_accuracy_from_csv(text, gt_label_arr, score_arr)

    print(f"Overall Accuracy: {accuracy}")
    for class_name, accuracy in class_accuracy.items():
        print(f"{class_name} Accuracy: {accuracy}")

    return

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
