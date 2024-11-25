import json
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
from PIL import Image

import clip
from eval_clip import *


# from waterbird_questions import generate_questions
# from llava_inference import llava_inference
def load_model(model, device, load_path):
    model_dict = torch.load(load_path)
    model.load_state_dict(model_dict['model']['state_dict'])
    model = model.to(device)
    return model


def main():
    text = ["waterbird", "landbird"]
    model_type = "ViT-B/32"

    scratch_dir = os.getenv("SCRATCH")
    run_dir = scratch_dir + "/pipeline_results0003"
    # os.mkdir(run_dir)

    # generate_questions("test", run_dir, per_class_count=100)
    # llava_inference(run_dir)

    print(torch.cuda.is_available())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load(model_type, device="cuda")
    model = load_model(model, "/scratch/nk3853/pipeline_results0000/Run0023/Train/best.tar", device)
    test_data = get_organized_dataset(scratch_dir + "/datasets", 'waterbirds', "test")
    dataset = CLIPDataloader(clip_transform= preprocess, clip_tokenizer=clip.tokenize, learning_data= test_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=2, shuffle=True)
    encodings = []
    label_to_num = {
        'waterbirdwater': 0,
        'waterbirdland': 1,
        'landbirdwater': 2,
        'landbirdland': 3
    }
    # for datapoint in tqdm(dataset):
    #     # object_text = datapoint['object_text']
    #     # bground_text = datapoint['bground_text']
    #     # obj_bg_concat_txt = datapoint['obj_bg_concat_txt']
    #     image = datapoint['image']
    #     encoding = {
    #         # 'object_text': model.encode_text(object_text).detach().cpu().numpy(),
    #         # 'bground_text': model.encode_text(bground_text).detach().cpu().numpy(),
    #         # 'image_text': model.encode_text(obj_bg_concat_txt).detach().cpu().numpy(),
    #         'image': model.encode_image(image.unsqueeze(dim=0)).detach().cpu().numpy(),
    #         'object_class': datapoint['label'],
    #         'bground_class': datapoint['place'],
    #         'label': label_to_num[datapoint['label'] + datapoint['place']],
    #         'class': 0 if datapoint['label'] == 'waterbird' else 1
    #     }

    #     encodings.append(encoding)
    # np.save(f'{run_dir}/image_encodings-small-all-test.npy', encodings)
    encodings = np.load(f'{run_dir}/image_encodings-small-finetuned-all-test.npy', allow_pickle=True)
    data = np.concatenate([encoding['image'] for encoding in encodings], axis=0)
    labels = [encoding['class'] for encoding in encodings]

    pca = PCA(n_components=50, random_state=42)
    data_pca = pca.fit_transform(data)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    data_2d = tsne.fit_transform(data_pca)
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, s=5)
    plt.colorbar(scatter, label='Labels')
    plt.title('t-SNE Visualization of 512-Dimensional Vectors')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.savefig("image_pretrained-clip-small-all-test2.png")

main()