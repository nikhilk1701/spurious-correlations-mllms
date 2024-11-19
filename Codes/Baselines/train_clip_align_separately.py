# Code to: Train CLIP with our method and losses but separately train image and text encoder
# Date Created: 10/29/2024
# Last Modified By: Shika

# Code to:
# Has all the code where we do decoupled training of text and image encoders
# This means, text and image encoder trained separately but with same data

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
from losses import *
from networks import *
import torch.nn.functional as F
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class Align_CLIP_Separately(nn.Module):

    def __init__(self, model_type, dataset, text_classes, bg_classes, img_dir, config):
        super(Align_CLIP_Separately, self).__init__() 

        self.config = config
        self.dataset = dataset
        self.text_classes = text_classes
        self.img_dir = img_dir
        self.model_type = model_type
        self.bg_classes = bg_classes

        # Added these for CLIP training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_type, device= self.device)

        if self.config.network_type == 'modified_clip':
            self.model = CLIPCombinedModified(self.model, layer_type=self.config.layer_type_for_modified_clip)

        # Get text features
        self.tokenized_text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in text_classes]).to(self.device)

        # For the optimizer, the text optimizer should only optimize everything except visual encoder
        text_optimizable_params = [param for name, param in self.model.named_parameters() if 'visual' not in name]
        visual_optimizable_params = [param for name, param in self.model.named_parameters() if 'visual' in name]
        if self.config.optimizer == 'adam':
            self.optimizer_text = Adam(text_optimizable_params, weight_decay=self.config.weight_decay, betas=(0.9,0.98), eps=1e-6, lr=self.config.learning_rate)
            self.optimizer_visual = Adam(visual_optimizable_params, weight_decay=self.config.weight_decay, betas=(0.9,0.98), eps=1e-6, lr=self.config.learning_rate)

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

        self.train_data = get_organized_dataset_auxillary(self.img_dir, self.dataset, "train", self.config.llava_out_dir)
        self.test_data = get_organized_dataset(self.img_dir, self.dataset, "test")

        train_dataset = CLIPDataloader(clip_transform= self.preprocess, clip_tokenizer=clip.tokenize, learning_data= self.train_data)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size, pin_memory=True, num_workers=self.config.num_gpu_workers, shuffle=True)

        # val_dataset = CLIPDataloader(clip_transform= self.preprocess, clip_tokenizer=clip.tokenize, learning_data= self.val_data)
        # self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size, pin_memory=True, num_workers=self.config.num_gpu_workers, shuffle=False)

        test_dataset = CLIPDataloader(clip_transform= self.preprocess, clip_tokenizer=clip.tokenize, learning_data= self.test_data)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.batch_size, pin_memory=True, num_workers=self.config.num_gpu_workers, shuffle=False)
        return

    # Freeze text and image encoders according to what we want when training
    # This is different from train_clip_align_simultaneously
    @staticmethod
    def weight_mode(model, freeze_what = "text"): # freeze_what = "text" or "image" or "both"
        if freeze_what == "image":
            for pname, param in model.named_parameters():
                if "visual" in pname:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
        
        elif freeze_what == "text":
            for pname, param in model.named_parametersparameters():
                if "visual" in pname:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        
        elif freeze_what == "both":
            for param in model.parameters():
                param.requires_grad_(False)
        return model

    # https://github.com/openai/CLIP/issues/83
    # Convert to fp32 for optimization 
    def convert_models_to_fp32(self, model):
        for p in model.parameters():
            p.data = p.data.float()
        return
    
    # To save the model then continue training
    def save_model_text(self, model, optimizer_text, best= False):

        # save the params which don't have visual in the name
        parameters = {name: param.data for name, param in model.named_parameters() if 'visual' not in name}

        model_ckpt_path = Path(self.config.results_dir) / 'Train'
        if not os.path.exists(model_ckpt_path):
            os.mkdir(model_ckpt_path)  
        if best:
            model_ckpt_path = os.path.join(model_ckpt_path, 'text_best.tar')
        else:
            model_ckpt_path = os.path.join(model_ckpt_path, 'text_latest.tar')
        save_dict = {'state_dict': parameters}
        save_opt_text = {'state_dict': optimizer_text.state_dict()}
        full_dict = {'model_text': save_dict, 'current_iteration_text': self.text_current_iteration, 'optimizer_text': save_opt_text}

        torch.save(full_dict, model_ckpt_path)

        return
    
    def save_model_image(self, model, optimizer_image, best= False):

        # save the params which have visual in the name
        parameters = {name: param.data for name, param in model.named_parameters() if 'visual' in name}

        model_ckpt_path = Path(self.config.results_dir) / 'Train'
        if not os.path.exists(model_ckpt_path):
            os.mkdir(model_ckpt_path)  
        if best:
            model_ckpt_path = os.path.join(model_ckpt_path, 'image_best.tar')
        else:
            model_ckpt_path = os.path.join(model_ckpt_path, 'image_latest.tar')
        save_dict = {'state_dict': parameters}
        save_opt_img = {'state_dict': optimizer_image.state_dict()}
        full_dict = {'model_text': save_dict, 'current_iteration_image': self.image_current_iteration, 'optimizer_image': save_opt_img}

        torch.save(full_dict, model_ckpt_path)

        return

    def load_model_text(self, load_path):
        model_dict = torch.load(load_path)
        self.model.load_state_dict(model_dict['model']['state_dict'], strict=False) # load into the full model but only the trained and saved parts
        self.optimizer_text.load_state_dict(model_dict['optimizer_text']['state_dict'])
        self.model = self.model.to(self.device)
        self.text_current_iteration = model_dict['current_iteration_text']
        return
    
    def load_model_image(self, load_path):
        model_dict = torch.load(load_path)
        self.model.load_state_dict(model_dict['model']['state_dict'], strict=False) # load into the full model but only the trained and saved parts
        self.optimizer_visual.load_state_dict(model_dict['optimizer_img']['state_dict'])
        self.model = self.model.to(self.device)
        self.image_current_iteration = model_dict['current_iteration_image']
        return
    

    # THIS HAS TO BE ADDED CAUSE OF THE THROWAWAY NETWORK CODE :(
    # This has to be used only while training. At inference, I have set the default to True in the network itself when calling encode_text
    def helper_encode_text(self, input_to_encode_text, throw_it= False):
        if self.config.network_type == 'modified_clip':
            logits_text = self.model.encode_text(input_to_encode_text, throw_it)
        elif self.config.network_type == 'clip':
            logits_text = self.model.encode_text(input_to_encode_text)
        return logits_text
        
    
    def train_text_encoder(self):

        text_train_loss = []
        self.text_current_iteration = 1
        self.init_dataloaders()
        iterator_model = iter(self.train_loader)

        self.test_dict['test_acc'] = {'acc_value': [], 'iter_no': []}

        start_iteration = 1
        total_iterations = int((self.config.train_text_epochs * len(self.train_loader)))
        test_iteration = max(1, int((self.config.test_epochs * len(self.train_loader))))

        self.model = self.weight_mode(self.model, freeze_what = "image")
        self.model.train()
        self.convert_models_to_fp32(self.model)

        loss_text = SupervisedContrastiveLoss(temperature= self.config.temperature, base_temperature= self.config.base_temperature, contrast_mode= "all")

        for iteration in tqdm(range(start_iteration, total_iterations + 1)):
            # currently from dataloader we get:
            # 1. image (anchor) pixels
            # 2. text label for image (indicates whether landbird or waterbird in text) (landbird, waterbird) (gt rom dataset)
            # 3. text related to class in image (+ve) obj description text
            # 4. text about the background (not related to image or text of class in image) (-ve in text space)
            # 5. background text label (indicates land or water) (gt from dataset)

            sampled_batch, iterator_model = self.get_next_train_batch(self.train_loader, iterator_model)
            images = sampled_batch['image']
            gt_label_image_text = sampled_batch['label']
            obj_text_description = sampled_batch['object_text'].squeeze()
            background_text_description = sampled_batch['bground_text']
            gt_label_background = sampled_batch['place']

            
            # in this case use only the object text description
            if self.config.background_consider == False:
                # convert gt_label_img_text_vector to 0 or 1 based on classname waterbird =0, landbird = 1
                class_to_index = {cls: idx for idx, cls in enumerate(self.text_classes)}
                gt_og_input_text = torch.tensor([class_to_index[label] for label in self.text_classes], device=self.device).unsqueeze(1) # 2
                gt_label_batch = torch.tensor([class_to_index[label] for label in gt_label_image_text], device=self.device).unsqueeze(1) # bs

                obj_text_description = obj_text_description.to(self.device) # bs
                # now concatenate the "waterbird" and "landbird" text vectors to text_description and also create it's corresponding labels
                texts = torch.cat((self.tokenized_text_inputs, obj_text_description), dim=0).to(self.device) # bs+2
                gt_label_text = torch.cat((gt_og_input_text, gt_label_batch), dim=0).to(self.device) # bs+2

                # This and the below line in else condition is the only case where throw_it will be False. This is because the throw_away head used only while training text encoder.
                logits_text = self.helper_encode_text(texts, throw_it=False) # bs+2 * 1024 or bs+2*128 if throwaway
                text_loss_term = loss_text(logits_text, gt_label_text)

            
            # In this case the background is considered but the background text is negative to both the text classes
            else:
                # also consider the background text as 2 diferent classes according to water and land
                class_to_index = {cls: idx for idx, cls in enumerate(self.text_classes + self.bg_classes)} # so 4 classes now
                gt_og_input_text = torch.tensor([class_to_index[label] for label in self.text_classes], device=self.device).unsqueeze(1) # 2
                gt_label_batch_text_desc = torch.tensor([class_to_index[label] for label in gt_label_image_text], device=self.device).unsqueeze(1) # bs
                gt_label_batch_background = torch.tensor([class_to_index[label] for label in gt_label_background], device=self.device).unsqueeze(1) # bs

                obj_text_description = obj_text_description.to(self.device) # bs
                background_text_description = background_text_description.to(self.device) # bs
                texts = torch.cat((self.tokenized_text_inputs, obj_text_description, background_text_description), dim=0).to(self.device) # 2bs+2
                gt_label_text = torch.cat((gt_og_input_text, gt_label_batch_text_desc, gt_label_batch_background), dim=0).to(self.device) # 2bs+2

                # throw away is False here
                logits_text = self.helper_encode_text(texts, throw_it= False) # 2bs+2 * 1024 or 2bs+2 *128 if throwaway
                text_loss_term = loss_text(logits_text, gt_label_text)

            
            self.optimizer_text.zero_grad()
            text_loss_term.backward()
            self.optimizer_text.step()

            text_train_loss.append(text_loss_term.item())
            text_loss_dict = {'loss': text_train_loss[-1], 'iteration': self.text_current_iteration}
            self.logger.add_scalar(f'TrainLossText', text_loss_dict['loss'], text_loss_dict['iteration'])

            per_sample_loss = text_train_loss[-1] / self.config.batch_size
            print(f'Iteration {iteration} of text side training done with per {per_sample_loss:0.4f}.')

            if iteration % test_iteration == 0 or iteration == total_iterations:
                self.test_dict['test_acc']['iter_no'].append(self.text_current_iteration)

                # saves the model according to test frequency
                print("Saving model before testing")
                self.save_model_text(self.model, self.optimizer_text, best=False)
                self.test_during_train(iteration= self.text_current_iteration, model_flag = "text") # saves the model again if it's best here

                self.model = self.weight_mode(self.model, freeze_what = "image")
                self.model.train()
                self.convert_models_to_fp32(self.model)

            self.text_current_iteration += 1

            del sampled_batch, images, texts
            torch.cuda.empty_cache()

        return

    def train_image_encoder(self):
        image_train_loss = []
        self.image_current_iteration = 1
        self.init_dataloaders()
        iterator_model = iter(self.train_loader)

        self.test_dict['test_acc'] = {'acc_value': [], 'iter_no': []}

        start_iteration = 1
        total_iterations = int((self.config.train_image_epochs * len(self.train_loader)))
        test_iteration = max(1, int((self.config.test_epochs * len(self.train_loader))))

        self.model = self.weight_mode(self.model, freeze_what = "text")
        self.model.train()
        self.convert_models_to_fp32(self.model)

        loss_img = BatchedSupervisedContrastiveLoss(temperature= self.config.temperature, base_temperature= self.config.base_temperature, contrast_mode= "one")

        for iteration in tqdm(range(start_iteration, total_iterations + 1)):
            # currently from dataloader we get:
            # 1. image (anchor) pixels
            # 2. text label for image (indicates whether landbird or waterbird in text) (landbird, waterbird) (gt rom dataset)
            # 3. text related to class in image (+ve) obj description text
            # 4. text about the background (not related to image or text of class in image) (-ve in text space)
            # 5. background text label (indicates land or water) (gt from dataset)

            sampled_batch, iterator_model = self.get_next_train_batch(self.train_loader, iterator_model)
            images = sampled_batch['image']
            gt_label_image_text = sampled_batch['label']
            obj_text_description = sampled_batch['object_text'].squeeze()
            background_text_description = sampled_batch['bground_text'] # we don't consider this in case of image training
            gt_label_background = sampled_batch['place'] # we don't consider this in case of image training

            # convert gt_label_img_text_vector to 0 or 1 based on classname waterbird =0, landbird = 1
            class_to_index = {cls: idx for idx, cls in enumerate(self.text_classes)}
            gt_og_input_text = torch.tensor([class_to_index[label] for label in self.text_classes], device=self.device).unsqueeze(1) # 2
            gt_label_batch = torch.tensor([class_to_index[label] for label in gt_label_image_text], device=self.device).unsqueeze(1) # bs

            # convert images to column vector
            images = images.to(self.device) # bs
            logits_image = self.model.encode_image(images) # bs * 1024
            logits_image = logits_image.unsqueeze(1) # bs * 1 * 1024
            gt_label_image = gt_label_batch.unsqueeze(1) # bs * 1

            # Here in this case we include the text classes "waterbird" and "landbird" in the image training
            if self.config.include_classtext_in_image_training == True:

                obj_text_description = obj_text_description.to(self.device) # bs
                # now concatenate the "waterbird" and "landbird" text vectors to text_description and also create it's corresponding labels
                texts = torch.cat((self.tokenized_text_inputs, obj_text_description), dim=0).to(self.device) # bs+2
                gt_label_text = torch.cat((gt_og_input_text, gt_label_batch), dim=0).to(self.device) # bs+2
                gt_label_text = gt_label_text.unsqueeze(0) # 1 * (bs+2)
                gt_label_text = gt_label_text.repeat(images.shape[0], 1, 1) # b * (bs + 2)

                # As we need a 1024 representation layer again, throw_it= True
                logits_text = self.helper_encode_text(texts, throw_it= True) # bs+2 * 1024
                logits_text = logits_text.unsqueeze(0) # 1 * (bs+2) * 1024
                logits_text = logits_text.repeat(images.shape[0], 1, 1) # bs * (bs + 2) * 1024

            # In this case we don't
            else:
                # As we need a 1024 representation layer again, throw_it= True
                logits_text = self.helper_encode_text(obj_text_description, throw_it= True).unsqueeze(0) # 1 * bs * 1024
                logits_text = logits_text.repeat(logits_text.shape[1], 1, 1) # bs * bs* 1024
                gt_label_text = gt_label_batch.unsqueeze(0) #  1 * bs
                gt_label_text = gt_label_text.repeat(gt_label_text.shape[1], 1, 1)

            # concatenate the image and text features and their label vectors too to get 16*19*1024 or 16*17*1024
            logits = torch.cat((logits_image, logits_text), dim=1) # bs* bs+3 * 1024 or bs* bs+1 * 1024
            labels_image_text = torch.cat((gt_label_image, gt_label_text), dim=1) # bs* bs+3 or bs* bs+1
            image_loss_term = loss_img(logits, labels_image_text)
            

            self.optimizer_visual.zero_grad()
            image_loss_term.backward()
            self.optimizer_visual.step()

            image_train_loss.append(image_loss_term.item())
            image_loss_dict = {'loss': image_train_loss[-1], 'iteration': self.image_current_iteration}
            self.logger.add_scalar(f'TrainLossImage', image_loss_dict['loss'], image_loss_dict['iteration'])

            per_sample_loss = image_train_loss[-1] / self.config.batch_size
            print(f'Iteration {iteration} of image side training done with per closs {per_sample_loss:0.4f}.')

            if iteration % test_iteration == 0 or iteration == total_iterations:
                self.test_dict['test_acc']['iter_no'].append(self.image_current_iteration)

                # saves the model according to test frequency
                print("Saving model before testing")
                self.save_model_image(self.model, self.optimizer_visual, best=False)
                self.test_during_train(iteration= self.image_current_iteration, model_flag = "image") # saves the model again if it's best here

                self.model = self.weight_mode(self.model, freeze_what = "text")
                self.model.train()
                self.convert_models_to_fp32(self.model)

            self.image_current_iteration += 1

            del sampled_batch, images, texts
            torch.cuda.empty_cache()

        return
    

    def test_during_train(self, iteration, model_flag):

        self.test_dict['csv'] = {'Image_Name': [], 'GT': [], f'pred{iteration:04d}': []}
        img_name, gt_labels, scores = clip_inference(self.model, self.tokenized_text_inputs, self.test_loader, self.device)
        
        self.test_dict['csv'][f'pred{iteration:04}'] = scores
        self.test_dict['csv']['Image_Name'] = img_name
        self.test_dict['csv']['GT'] = gt_labels

        accuracy, class_accuracy = calculate_class_pred_accuracy(self.text_classes, gt_labels, scores)

        self.test_dict['csv'][f'pred{iteration:04}'].append(accuracy)
        self.test_dict['csv']['Image_Name'].append('Accuracy')
        self.test_dict['csv']['GT'].append(1.0)

        del img_name, gt_labels, scores

        details_path = os.path.join(self.config.results_dir, 'details.txt')
        logging.basicConfig(filename=details_path, filemode='a', level=logging.DEBUG, format='')

        print(f"Overall Accuracy: {accuracy}")
        logging.info(f"Overall Accuracy: {accuracy}")

        for class_name, accuracy in class_accuracy.items():
            print(f"{class_name} Accuracy: {accuracy}")
            logging.info(f"{class_name} Accuracy: {accuracy}")

        # Saving test performance to disk
        if not os.path.exists((Path(self.config.results_dir) / 'Test').as_posix()):
            os.mkdir((Path(self.config.results_dir) / 'Test').as_posix())

        save_dir = (Path(self.config.results_dir) / f'Test/predictions.csv').as_posix()

        if self.save_flag:
            df = pd.DataFrame.from_dict(self.test_dict['csv'])
            df.to_csv(save_dir, index=False)
        else:
            df1 = pd.read_csv(save_dir)
            df1[f'pred{iteration:04}'] = self.test_dict['csv'][f'pred{iteration:04}']
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
        if model_flag == "image":
            if len(self.test_acc['acc']) > 0:
                if accuracy > max(self.test_acc['acc']):
                    self.save_model_image(self.model, self.optimizer_visual, best=True)

        elif model_flag == "text":
            if len(self.test_acc['acc']) > 0:
                if accuracy > max(self.test_acc['acc']):
                    self.save_model_image(self.model, self.optimizer_text, best=True)

        self.test_acc['acc'].append(accuracy)
        self.test_acc['iteration'].append(iteration)

        return
    
def configuration_params():
    parser = argparse.ArgumentParser()
    scratch_dir = os.getenv("SCRATCH")
    results_dir = scratch_dir + '/results/Train'

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_text_epochs', type=int, default=5)
    parser.add_argument('--train_image_epochs', type=int, default=5)
    parser.add_argument('--test_epochs', type=int, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_gpu_workers', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default=results_dir)
    parser.add_argument('--model_type', type=str, default='ViT-B/32', choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'SGD'])
    parser.add_argument('--network_type', type=str, default= 'clip', choices=['clip', 'modified_clip'])
    parser.add_argument('--layer_type_for_modified_clip', type=str, default= 'linear', choices=['mlp', 'linear', 'throwaway'])
    
    parser.add_argument('--background_consider', type=bool, default=False)
    parser.add_argument('--include_classtext_in_image_training', type=bool, default=False)

    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--base_temperature', type=float, default=0.07)
    parser.add_argument('--llava_out_dir', type=str, default=f'{scratch_dir}/pipeline_results0000')

    config = parser.parse_args()
    return config


def train_clip_align_separately(run_dir, learning_rate, include_classtext_in_image_training = False, background_consider=False):
    config = configuration_params()
    config.llava_out_dir = run_dir
    config.results_dir = run_dir
    config.learning_rate = learning_rate
    config.include_classtext_in_image_training = include_classtext_in_image_training
    config.background_consider = background_consider
    scratch_dir = os.getenv("SCRATCH")
    img_dir = scratch_dir + '/datasets'
    model = Align_CLIP_Separately(model_type=config.model_type, dataset='waterbirds', text_classes=['waterbird', 'landbird'], bg_classes= ['water', 'land'], img_dir=img_dir, config=config)

    model.train_text_encoder()
    model.train_image_encoder()

    return


if __name__ == '__main__':
    print('Program started at ' +
          datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        train_clip_align_separately()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' +
          datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))