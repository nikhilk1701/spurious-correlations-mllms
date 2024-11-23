import os
import json
from dataloader import *

image_dir = "/scratch/nk3853/datasets/Waterbirds"

"""
    generates prompts for llava to generate the auxillary text
    currently limits to 10 images of each kind [waterbird, landbird] X [landbackground, waterbackground] for easy testing

    run using
        ''echo > questions.jsonl; python waterbird_questions.py''
"""
def generate_questions(dataset_split, results_dir, per_class_count = 500):
    text = ["waterbird", "landbird"]
    dataset_name = "waterbirds"
    scratch_dir = os.getenv("SCRATCH")
    img_dir = scratch_dir + '/datasets'

    # getting just the train set here 
    dataset = get_organized_dataset(base_dataset_path=Path(img_dir), dataset_name=dataset_name, dataset_split=dataset_split)

    questions_object = []
    questions_bground = []
    question_id = 1
    wblb = 0
    wbwb = 0
    lblb = 0
    lbwb = 0

    for datapoint in dataset:
        image_class = datapoint['label']
        background = datapoint['place']

        question_bground = {}
        question_object  = {}

        question = {}
        question["question_id"] = question_id
        question["image"] = datapoint["name"]
        question["label"] = image_class

        question_object = question.copy()
        question_bground = question.copy()
        # question["text"] = f"Describe the {image_class} in the image, ignore the {background}. Include what visual features it has common with other birds of the same class. Be concise."
        # question["text"] = f"I think that this is a landbird. If I am wrong, tell me why."            
        question_bground["text"] = "Describe the background in the image, ignore the bird. Be concise."
        question_object["text"] = "Identify the bird in the image and describe distinguishing features of the bird in the image. Ignore the background, be concise."
        
        # indicates that it is the entire dataset as -1 indicates end right
        if per_class_count == -1: 
            questions_bground.append(question_bground)
            questions_object.append(question_object)
            question_id += 1

        # When we specify some balanced subset of the dataset
        else:
            if image_class == "waterbird" and background == "land" and wblb < per_class_count:
                questions_bground.append(question_bground)
                questions_object.append(question_object)
                wblb += 1
                question_id += 1

            if image_class == "waterbird" and background == "water" and wbwb < per_class_count:
                questions_bground.append(question_bground)
                questions_object.append(question_object)
                wbwb += 1
                question_id += 1
            
            if image_class == "landbird" and background == "land" and lblb < per_class_count:
                questions_bground.append(question_bground)
                questions_object.append(question_object)
                lblb += 1
                question_id += 1
            
            if image_class == "landbird" and background == "water" and lbwb < per_class_count:
                questions_bground.append(question_bground)
                questions_object.append(question_object)
                lbwb += 1
                question_id += 1

    with open(f"{results_dir}/questions_bground.jsonl", 'a') as questions_file:
        for question in questions_bground:
            questions_file.write(json.dumps(question) + "\n")

    with open(f"{results_dir}/questions_object.jsonl", 'a') as questions_file:
        for question in questions_object:
            questions_file.write(json.dumps(question) + "\n")

## question format for reference
# {"question_id": 248, "text": "I think that this is a landbird. If I am wrong, tell me why.", "image": "005.Crested_Auklet/Crested_Auklet_0047_794918.jpg"}
# {"question_id": 249, "text": "I think that this is a landbird. If I am wrong, tell me why.", "image": "005.Crested_Auklet/Crested_Auklet_0005_794922.jpg"}
# {"question_id": 250, "text": "I think that this is a landbird. If I am wrong, tell me why.", "image": "005.Crested_Auklet/Crested_Auklet_0019_794925.jpg"}

## answer format for reference
# {"question_id": 7, "prompt": "Describe the waterbird in the image, ignore the water.", "text": "The waterbird in the image is a large, brown bird with a long beak. It is standing on a rocky shore near the water, possibly on a beach.", "answer_id": "Vw5sSjtcvkzwcZk2csxisv", "model_id": "llava-v1.5-7b", "metadata": {}}
# {"question_id": 8, "prompt": "Describe the waterbird in the image, ignore the water.", "text": "The waterbird in the image is a large, brown, and white bird with a long beak. It appears to be a bird of prey, such as a pelican or a stork.", "answer_id": "K9owqrK8FAhSAfrVLimugh", "model_id": "llava-v1.5-7b", "metadata": {}}
# {"question_id": 9, "prompt": "Describe the waterbird in the image, ignore the water.", "text": "The waterbird in the image is a large bird with a brown body and wings. It is flying over a body of water, possibly a lake or a river.", "answer_id": "FiwDydR3EijRavtbG6hSki", "model_id": "llava-v1.5-7b", "metadata": {}}
# {"question_id": 10, "prompt": "Describe the waterbird in the image, ignore the water.", "text": "The waterbird in the image is a large black bird with a white head. It is sitting on a branch or a nest, possibly on a cliff or a rock.", "answer_id": "fHRAktxYzbnS75hSRVqLsa", "model_id": "llava-v1.5-7b", "metadata": {}}
