# Code to: Evaluate llava on the waterbirds classification task
# Date Created: 11/22/2024
# Last Modified By: Shika

import os
from data_read_utils import *
from dataloader import *
from llava.mm_utils import get_model_name_from_path
from llava.eval.model_vqa import eval_model
import traceback
import datetime
import time
import json


def generate_question_answers(text, results_dir, img_dir, dataset_name):

    # getting just the test set here 
    dataset = get_organized_dataset(base_dataset_path=Path(img_dir), dataset_name=dataset_name, dataset_split="test")

    questions_object = []
    question_id = 1

    for datapoint in dataset:
        image_class = datapoint['label']
        question_object  = {}

        question = {}
        question["question_id"] = question_id
        question["image"] = datapoint["name"]
        question["label"] = image_class

        question_object = question.copy()
        question_object["text"] = "Give one word to classify the bird in the image from the following classes: "
        for i in range(len(text)):
            if i == len(text) - 1:
                question_object["text"] += text[i]
            else:
                question_object["text"] += text[i] + ", "
        questions_object.append(question_object)
        question_id += 1

    with open(f"{results_dir}/questions_object.jsonl", 'a') as questions_file:
        for question in questions_object:
            questions_file.write(json.dumps(question) + "\n")
    
    return

def llava_inference(results_dir, image_folder):
    model_path = "liuhaotian/llava-v1.5-7b"

    object_qfile = f"{results_dir}/questions_object.jsonl"
    object_afile = f"{results_dir}/answers_object.jsonl"

    with open(object_afile, 'w') as oafp:
        pass

    args_object = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "conv_mode": None,
        "sep": ",",
        "temperature": 0.2,
        "top_p": 0.001,
        "num_beams": 5,
        "max_new_tokens": 77,
        "question_file": object_qfile,
        "answers_file": object_afile,
        "image_folder": image_folder,
        "num_chunks": 1,
        "conv_mode": "llava_v1",
        "chunk_idx": 0,
    })()
    eval_model(args_object)

    return

def read_results(results_dir, dataset_name, img_dir):
    dataset = get_dataset_list(base_dataset_path=Path(img_dir), dataset= dataset_name)
    dataset_dict = {}

    for datapoint in dataset:
        dataset_dict[datapoint['name']] = datapoint
    
    object_questions = results_dir + '/questions_object.jsonl'
    object_answers = results_dir + '/answers_object.jsonl'

    with open(object_answers, 'r') as afile, open(object_questions, 'r') as qfile:
        for (qline, aline) in zip(qfile, afile):
            ques = json.loads(qline)
            ans = json.loads(aline)
            dataset_dict[ques['image']]['object_text'] = ans['text']

    return [val for val in list(dataset_dict.values()) if 'object_text' in val]

def calculate_class_pred_accuracy(class_order, gt_labels, llava_text_pred):
    
    # Calculate overall accuracy    
    correct_predictions = sum(pred == gt for pred, gt in zip(llava_text_pred, gt_labels))
    accuracy = correct_predictions / len(gt_labels)

    # calculate accuracy per class
    class_accuracy = {}
    for class_name in class_order:
        class_indices = [i for i, gt in enumerate(gt_labels) if gt == class_name]
        if len(class_indices) == 0:
            continue
        class_gt_labels = [gt_labels[i] for i in class_indices]
        tmp = [llava_text_pred[i] for i in class_indices]
        correct_class_predictions = sum(pred == gt for pred, gt in zip(tmp, class_gt_labels))
        class_accuracy[class_name] = correct_class_predictions / len(class_indices)

    return accuracy, class_accuracy

def calculate_4class_pred_accuracy(dataset, llava_text_pred, class_order):
    total_counts = {
        "waterbird": { "land": 0, "water": 0, },
        "landbird": { "land": 0, "water": 0, },
    }
    correct_counts = {
        "waterbird": { "land": 0, "water": 0, },
        "landbird": { "land": 0, "water": 0, },
    }
    for datapoint, pred_class in zip(dataset, llava_text_pred):
        guess = pred_class

        gt_label = datapoint['label']
        gt_place = datapoint['place']

        if guess == gt_label:
            correct_counts[guess][gt_place] += 1
        
        total_counts[gt_label][gt_place] += 1
    
    accuracy = { }

    for gt_class in class_order:
        accuracy[gt_class] = {}
        for place in ["water", "land"]:
            if total_counts[gt_class][gt_place] != 0:
                accuracy[gt_class][place] = (correct_counts[gt_class][place] * 1.0) / total_counts[gt_class][gt_place]
    return accuracy



def main():
    text = ["waterbird", "landbird"]
    results_dir = "/scratch/sr7463/results"
    dataset_name = "waterbirds"
    scratch_dir = os.getenv("SCRATCH")
    img_dir = scratch_dir + '/datasets'
    image_folder = f"{img_dir}/Waterbirds"

    generate_question_answers(text, results_dir, img_dir, dataset_name)
    llava_inference(results_dir, image_folder)
    organized_dataset = read_results(results_dir, dataset_name, img_dir)

    # get "label" from organized dataset and "object_text" from organized dataset
    gt_labels = [datapoint['label'] for datapoint in organized_dataset]
    class_pred_llava = [datapoint['object_text'] for datapoint in organized_dataset]

    # convert all classes to lowercase
    gt_labels = [label.lower() for label in gt_labels]
    class_pred_llava = [label.lower() for label in class_pred_llava]

    accuracy, class_accuracy = calculate_class_pred_accuracy(text, gt_labels, class_pred_llava)
    subclass_accuracy = calculate_4class_pred_accuracy(organized_dataset, class_pred_llava, text)
    
    print(f"Overall Accuracy: {accuracy}")
    for class_name, class_acc in class_accuracy.items():
            print(f"{class_name} Accuracy: {class_acc}")
    for c_name, c_acc in subclass_accuracy.items():
        for subc_name, subc_acc in c_acc.items():
            print(f"{c_name} on {subc_name} background accuracy: {subc_acc}")

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