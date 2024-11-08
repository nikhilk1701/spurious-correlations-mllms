import datetime

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.model_vqa import eval_model

model_path = "liuhaotian/llava-v1.5-7b"
question_file = "./outputs/questions_object_test.jsonl"
image_folder = "/scratch/nk3853/datasets/Waterbirds"
answers_file = f"/scratch/nk3853/answers_object_test_{str(datetime.datetime.now())}.jsonl"

with open(answers_file, 'w') as fp:
    pass

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "conv_mode": None,
    "sep": ",",
    "temperature": 0.2,
    "top_p": 0.001,
    "num_beams": 5,
    "max_new_tokens": 77,
    "question_file": question_file,
    "answers_file": answers_file,
    "image_folder": image_folder,
    "num_chunks": 1,
    "conv_mode": "llava_v1",
    "chunk_idx": 0,
})()

eval_model(args)
