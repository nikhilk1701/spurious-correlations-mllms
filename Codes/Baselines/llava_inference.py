import os

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.model_vqa import eval_model

scratch_dir = os.getenv("SCRATCH")
model_path = "liuhaotian/llava-v1.5-7b"
image_folder = f"{scratch_dir}/datasets/Waterbirds"

def llava_inference(run_folder):
    object_qfile = f"{run_folder}/questions_object.jsonl"
    bground_qfile = f"{run_folder}/questions_bground.jsonl"

    object_afile = f"{run_folder}/answers_object.jsonl"
    bground_afile = f"{run_folder}/answers_bground.jsonl"

    with open(object_afile, 'w') as oafp, open(bground_afile, 'w') as bafp:
        pass

    args_bground = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "sep": ",",
        "temperature": 0.2,
        "top_p": 0.001,
        "num_beams": 5,
        "max_new_tokens": 77,
        "question_file": bground_qfile,
        "answers_file": bground_afile,
        "image_folder": image_folder,
        "num_chunks": 1,
        "conv_mode": "llava_v1",
        "chunk_idx": 0,
    })()
    eval_model(args_bground)

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