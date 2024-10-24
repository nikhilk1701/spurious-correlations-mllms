import os
import json
from dataloader import *


image_dir = "/scratch/nk3853/datasets/Waterbirds"

def generate_questions():
    text = ["waterbird", "landbird"]
    dataset_name = "waterbirds"
    scratch_dir = os.getenv("SCRATCH")
    img_dir = scratch_dir + '/datasets'

    dataset = get_organized_dataset(base_dataset_path=Path(img_dir), dataset_name=dataset_name, dataset_split='all_combined')

    questions = []
    question_id = 1
    for datapoint in dataset:
        question = {}
        # print(datapoint)
        question["question_id"] = question_id
        image_class = datapoint['label']
        background = datapoint['place']
        # question["text"] = f"Describe the {image_class} in the image, ignore the {background}. Include what visual features it has common with other birds of the same class. Be concise."
        question["text"] = f"I think that this is a landbird. If I am wrong, tell me why."
        question["image"] = datapoint["name"]
        questions.append(question)

        question_id += 1


    print(questions)

    with open('questions.jsonl', 'a') as questions_file:
        for question in questions:
            questions_file.write(json.dumps(question) + "\n")

generate_questions()

## question format for reference
# {"question_id": 248, "text": "I think that this is a landbird. If I am wrong, tell me why.", "image": "005.Crested_Auklet/Crested_Auklet_0047_794918.jpg"}
# {"question_id": 249, "text": "I think that this is a landbird. If I am wrong, tell me why.", "image": "005.Crested_Auklet/Crested_Auklet_0005_794922.jpg"}
# {"question_id": 250, "text": "I think that this is a landbird. If I am wrong, tell me why.", "image": "005.Crested_Auklet/Crested_Auklet_0019_794925.jpg"}

## answer format for reference
# {"question_id": 7, "prompt": "Describe the waterbird in the image, ignore the water.", "text": "The waterbird in the image is a large, brown bird with a long beak. It is standing on a rocky shore near the water, possibly on a beach.", "answer_id": "Vw5sSjtcvkzwcZk2csxisv", "model_id": "llava-v1.5-7b", "metadata": {}}
# {"question_id": 8, "prompt": "Describe the waterbird in the image, ignore the water.", "text": "The waterbird in the image is a large, brown, and white bird with a long beak. It appears to be a bird of prey, such as a pelican or a stork.", "answer_id": "K9owqrK8FAhSAfrVLimugh", "model_id": "llava-v1.5-7b", "metadata": {}}
# {"question_id": 9, "prompt": "Describe the waterbird in the image, ignore the water.", "text": "The waterbird in the image is a large bird with a brown body and wings. It is flying over a body of water, possibly a lake or a river.", "answer_id": "FiwDydR3EijRavtbG6hSki", "model_id": "llava-v1.5-7b", "metadata": {}}
# {"question_id": 10, "prompt": "Describe the waterbird in the image, ignore the water.", "text": "The waterbird in the image is a large black bird with a white head. It is sitting on a branch or a nest, possibly on a cliff or a rock.", "answer_id": "fHRAktxYzbnS75hSRVqLsa", "model_id": "llava-v1.5-7b", "metadata": {}}
