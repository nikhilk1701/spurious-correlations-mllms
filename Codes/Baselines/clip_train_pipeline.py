import os

from waterbird_questions import generate_questions
from llava_inference import llava_inference

scratch_dir = os.getenv("SCRATCH")
run_dir = scratch_dir + "/pipeline_results0000"
os.mkdir(run_dir)

generate_questions("train", run_dir, per_class_count=500)
llava_inference(run_dir)

from train_clip_align_simultaneously import train_clip_align_simultaneously
train_clip_align_simultaneously(run_dir, mode='text_image_concat_no_bg', background_consider=True, learning_rate=1e-6, include_classtext_in_image_training=False)
