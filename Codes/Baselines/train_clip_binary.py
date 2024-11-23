# Code to: Train CLIP on binary classification dataset like waterbirds
# Date Created: 10/13/24
# Last Modified By: Shika

import os
import argparse
import traceback
import datetime
import time
import json
import logging
from tqdm import tqdm
import torch.utils.data
import torch
import clip
import pandas as pd
import numpy as np
from pathlib import Path
import torch.nn as nn
from matplotlib import pyplot
from torch.optim import Adam, AdamW
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from dataloader import *
from eval_clip import *
import torch.nn.functional as F
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class Train_CLIP_Binary(nn.Module):

    def __init__(self, model_type, dataset, text_classes, img_dir, config):
        super(Train_CLIP_Binary, self).__init__() 

        self.config = config
        self.dataset = dataset
        self.text_classes = text_classes
        self.img_dir = img_dir
        self.model_type = model_type

        # Added these for CLIP training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_type, device= self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])

        # Get text features
        self.tokenized_text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in text_classes]).to(self.device)

        if self.config.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), weight_decay=self.config.weight_decay, betas=(0.9,0.98), eps=1e-6, lr=self.config.learning_rate)

        self.test_dict = {}
        self.test_acc = {'iteration': [], 'acc': []}

        # Setting up results folder
        run_number = len(os.listdir(self.config.results_dir))
        self.curr_result_dir = os.path.join(
            config.results_dir, f'Run{run_number:04}')
        if not os.path.exists(self.curr_result_dir):
            os.mkdir(self.curr_result_dir)
        self.config.results_dir = self.curr_result_dir

        # Dumping config details to folder
        config_details_path = os.path.join(
            self.config.results_dir, 'config_details.json')
        json_object = json.dumps(self.config.__dict__, indent=4)
        with open(config_details_path, "w") as outfile:
            outfile.write(json_object)

        self.logger = SummaryWriter((Path(self.curr_result_dir) / 'Logs').as_posix())
        self.save_flag = True

        return

    @staticmethod
    def get_next_train_batch(dataloader, iterator):
        try:
            next_batch = next(iterator)
        except StopIteration:
            print("Stop iteration encountered.")
            iterator = iter(dataloader)
            next_batch = next(iterator)
        return next_batch, iterator

    # Initialize dataloaders
    def init_dataloaders(self):

        self.train_data = get_organized_dataset(self.img_dir, self.dataset, "train")
        self.test_data = get_organized_dataset(self.img_dir, self.dataset, "test")

        train_dataset = CLIPDataloader(clip_transform= self.preprocess, clip_tokenizer=clip.tokenize, learning_data= self.train_data)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size, pin_memory=True, num_workers=self.config.num_gpu_workers, shuffle=True)

        # val_dataset = CLIPDataloader(clip_transform= self.preprocess, clip_tokenizer=clip.tokenize, learning_data= self.val_data)
        # self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size, pin_memory=True, num_workers=self.config.num_gpu_workers, shuffle=False)

        test_dataset = CLIPDataloader(clip_transform= self.preprocess, clip_tokenizer=clip.tokenize, learning_data= self.test_data)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.batch_size, pin_memory=True, num_workers=self.config.num_gpu_workers, shuffle=False)
        return

    # Makes a model's weights trainable/frozen
    @staticmethod
    def weight_mode(model, trainable=True):
        for param in model.parameters():
            if trainable:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        return model

    # https://github.com/openai/CLIP/issues/83
    # Convert to fp32 for optimization 
    def convert_models_to_fp32(self, model):
        for p in model.parameters():
            p.data = p.data.float()
        return

    # @staticmethod
    # def update_learning_rate(optimizer, factor):
    #     for group in optimizer.param_groups:
    #         group['lr'] *= factor
    #     return
    
    # To save the model then continue training
    def save_model(self, model, optimizer, best= False):
        model_ckpt_path = Path(self.config.results_dir) / 'Train'
        if not os.path.exists(model_ckpt_path):
            os.mkdir(model_ckpt_path)  
        if best:
            model_ckpt_path = os.path.join(model_ckpt_path, 'best.tar')
        else:
            model_ckpt_path = os.path.join(model_ckpt_path, 'latest.tar')
        save_dict = {'state_dict': model.state_dict()}
        save_opt = {'state_dict': optimizer.state_dict()}
        full_dict = {'model': save_dict, 'current_iteration': self.current_iteration, 'optimizer': save_opt}

        torch.save(full_dict, model_ckpt_path)

        return

    def load_model(self, load_path):
        model_dict = torch.load(load_path)
        self.model.load_state_dict(model_dict['model']['state_dict'])
        self.optimizer.load_state_dict(model_dict['optimizer']['state_dict'])
        self.model = self.model.to(self.device)
        self.current_iteration = model_dict['current_iteration']
        return

    def train_model(self):
        train_loss = []
        self.current_iteration = 1
        self.init_dataloaders()
        iterator_model = iter(self.train_loader)

        self.test_dict['test_acc'] = {'acc_value': [], 'iter_no': []}

        start_iteration = 1
        total_iterations = int((self.config.epochs * len(self.train_loader)))
        test_iteration = max(1,int((self.config.test_epochs * len(self.train_loader))))

        if self.config.resume_training == True:
            self.load_model(self.config.resume_model_path)
            start_iteration = self.current_iteration + 1

        self.model = self.weight_mode(self.model, trainable=True)
        self.model.train()
        self.convert_models_to_fp32(self.model)

        loss_img = nn.BCEWithLogitsLoss()
        loss_text = nn.CrossEntropyLoss()

        for iteration in tqdm(range(start_iteration, total_iterations + 1)):

            sampled_batch, iterator_model = self.get_next_train_batch(self.train_loader, iterator_model)
            images = sampled_batch['image']
            names = sampled_batch['name']
            labels = sampled_batch['label']

            images = images.to(self.device)
            texts = self.tokenized_text_inputs.to(self.device)

            # In the CLIP paper, they have an n,n matrix of logits. That is because they have same number of image and text features. 
            # Thus they have an image-class mathcing loss and a text-class matching loss. 
            # In our case, we have 32 image samples, but only 2 text samples.
            # Since this is binary classification task, we modify it a little bit.
            logits_per_image, logits_per_text = self.model(images, texts) # Shape [32,2]
            # Question: why is the just the first column selected
            selected_logits = logits_per_image[:, 0].unsqueeze(1)  # Now shape: [32, 1]
            # getting gt for image side training and text side training
            class_to_index = {cls: idx for idx, cls in enumerate(self.text_classes)}
            ground_truth_img = torch.tensor([class_to_index[label] for label in labels], device=self.device).unsqueeze(1)
            image_loss = loss_img(selected_logits, ground_truth_img.float())
            
            if self.config.mode == 'only_image':
                loss = image_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.append(loss.item())
            loss_dict = {'loss': train_loss[-1], 'iteration': self.current_iteration}
            self.logger.add_scalar(f'TrainLoss', loss_dict['loss'], loss_dict['iteration'])

            per_sample_loss = train_loss[-1] / self.config.batch_size
            print(f'Iteration {iteration} done with per sample loss {per_sample_loss:0.4f}.')

            if iteration % test_iteration == 0 or iteration == total_iterations:
                self.test_dict['test_acc']['iter_no'].append(self.current_iteration)

                # saves the model according to test frequency
                print("Saving model before testing")
                self.save_model(self.model, self.optimizer, best=False)
                self.test_during_train() # saves the model again if it's best here

                self.model = self.weight_mode(self.model, trainable=True)
                self.model.train()
                self.convert_models_to_fp32(self.model)

            self.current_iteration += 1

            del sampled_batch, images, texts, logits_per_image, logits_per_text
            torch.cuda.empty_cache()

        return

    def test_during_train(self):

        self.test_dict['csv'] = {'Image_Name': [], 'GT': [], f'pred{self.current_iteration:04d}': []}
        img_name, gt_labels, scores = clip_inference(self.model, self.tokenized_text_inputs, self.test_loader, self.device, binary = True)
        
        self.test_dict['csv'][f'pred{self.current_iteration:04}'] = scores
        self.test_dict['csv']['Image_Name'] = img_name
        self.test_dict['csv']['GT'] = gt_labels

        accuracy, class_accuracy = calculate_class_pred_accuracy(self.text_classes, gt_labels, scores, binary = True)

        self.test_dict['csv'][f'pred{self.current_iteration:04}'].append(accuracy)
        self.test_dict['csv']['Image_Name'].append('Accuracy')
        self.test_dict['csv']['GT'].append(1.0)

        del img_name, gt_labels, scores

        details_path = os.path.join(self.config.results_dir, 'details.txt')
        logging.basicConfig(filename=details_path, filemode='a', level=logging.DEBUG, format='')

        print(f"Overall Accuracy: {accuracy}")
        logging.info(f"Overall Accuracy: {accuracy}")

        for class_name, class_acc in class_accuracy.items():
            print(f"{class_name} Accuracy: {class_acc}")
            logging.info(f"{class_name} Accuracy: {class_acc}")

        # Saving test performance to disk
        if not os.path.exists((Path(self.config.results_dir) / 'Test').as_posix()):
            os.mkdir((Path(self.config.results_dir) / 'Test').as_posix())

        save_dir = (Path(self.config.results_dir) / f'Test/predictions.csv').as_posix()

        if self.save_flag:
            df = pd.DataFrame.from_dict(self.test_dict['csv'])
            df.to_csv(save_dir, index=False)
        else:
            df1 = pd.read_csv(save_dir)
            df1[f'pred{self.current_iteration:04}'] = self.test_dict['csv'][f'pred{self.current_iteration:04}']
            df1.to_csv(save_dir, index=False)

        # So test_dict[test_srocc] looks like {[srocc1 srocc2] [iter 1 2]}
        self.test_dict['test_acc']['acc_value'].append(accuracy)

        # Saving the test performance vs cycles
        pyplot.figure(1)
        pyplot.plot(self.test_dict['test_acc']['iter_no'], self.test_dict['test_acc']['acc_value'])
        pyplot.grid()
        pyplot.xlabel('Training Iteration')
        pyplot.ylabel('Accuracy')
        pyplot.savefig(Path(self.config.results_dir) / f'Test/test.png')

        self.save_flag = False

        # Saving the model if it's the best model
        if len(self.test_acc['acc']) > 0:
            if accuracy > max(self.test_acc['acc']):
                self.save_model(self.model, self.optimizer, best=True)

        self.test_acc['acc'].append(accuracy)
        self.test_acc['iteration'].append(self.current_iteration)

        return
    
def configuration_params():
    parser = argparse.ArgumentParser()
    results_dir =  os.getenv("SCRATCH") + '/results/Train'

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--test_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-7)
    parser.add_argument('--num_gpu_workers', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default=results_dir)
    parser.add_argument('--model_type', type=str, default='ViT-B/32', choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'SGD'])
    parser.add_argument('--mode', type=str, default='only_image', choices=['only_image'])
    config = parser.parse_args()
    return config


def main():
    config = configuration_params()
    img_dir = os.getenv("SCRATCH") + '/datasets'
    model = Train_CLIP_Binary(model_type=config.model_type, dataset='waterbirds', text_classes=['waterbird', 'landbird'], img_dir=img_dir, config=config)
    model.train_model()

    return


if __name__ == '__main__':
    print('Program started at ' +
          datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' +
          datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
