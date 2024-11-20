# Code to: Train CLIP with our method and losses but simultaneously
# Date Created: 10/29/2024
# Last Modified By: Shika

# Code to:
# Has all the code where we don't do decoupled training of text and image encoders
# This means, text and image encoder trained at the same time with same data

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

class Align_CLIP(nn.Module):

    def __init__(self, model_type, dataset, text_classes, bg_classes, img_dir, config):
        super(Align_CLIP, self).__init__() 

        self.config = config
        self.dataset = dataset
        self.text_classes = text_classes
        self.img_dir = img_dir
        self.model_type = model_type

        # Added these for CLIP training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_type, device= self.device)

        if self.config.network_type == 'modified_clip':
            self.model = CLIPCombinedModified(self.model, layer_type=self.config.layer_type_for_modified_clip)

        # Get text features
        self.tokenized_text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in text_classes]).to(self.device)
        self.bg_classes = bg_classes

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

        self.train_data = get_organized_dataset_auxillary(self.img_dir, self.dataset, "train", self.config.llava_out_dir)
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
    def weight_mode(model, trainable=True, probe=False, network_type='clip'):
        if network_type == 'modified_clip' and probe:
            # print([name for name, _ in model.named_parameters()])
            for name, param in model.named_parameters():
                if trainable and ("projection_head" in name):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        else:
            for _, param in model.named_parameters():
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
        test_iteration = max(1, int((self.config.test_epochs * len(self.train_loader))))
        if self.config.resume_training == True:
            self.load_model(self.config.resume_model_path)
            start_iteration = self.current_iteration + 1

        self.model = self.weight_mode(self.model, trainable=True, probe=self.config.probe, network_type=self.config.network_type)
        self.model.train()
        self.convert_models_to_fp32(self.model)

        if self.config.mode == 'text_image_concat_no_bg':
            loss_total = SupervisedContrastiveLoss(temperature= self.config.temperature, base_temperature= self.config.base_temperature, contrast_mode= "all")
        elif self.config.mode == 'separate_text_image':
            loss_img = BatchedSupervisedContrastiveLoss(temperature= self.config.temperature, base_temperature= self.config.base_temperature, contrast_mode= "one")
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

            if self.config.background_consider == False:
                # convert gt_label_img_text_vector to 0 or 1 based on classname waterbird =0, landbird = 1
                class_to_index = {cls: idx for idx, cls in enumerate(self.text_classes)}
                gt_og_input_text = torch.tensor([class_to_index[label] for label in self.text_classes], device=self.device).unsqueeze(1) # 2
                gt_label_batch = torch.tensor([class_to_index[label] for label in gt_label_image_text], device=self.device).unsqueeze(1) # bs
            else:
                # also consider the background text as 2 diferent classes according to water and land
                class_to_index = {cls: idx for idx, cls in enumerate(self.text_classes + self.bg_classes)} # so 4 classes now
                gt_og_input_text = torch.tensor([class_to_index[label] for label in self.text_classes], device=self.device).unsqueeze(1) # 2
                gt_label_batch_text_desc = torch.tensor([class_to_index[label] for label in gt_label_image_text], device=self.device).unsqueeze(1) # bs
                gt_label_batch_background = torch.tensor([class_to_index[label] for label in gt_label_background], device=self.device).unsqueeze(1) # bs


            if self.config.mode == 'text_image_concat_no_bg':
                images = images.to(self.device) # bs
                logits_image = self.model.encode_image(images) # bs * 1024
                
                obj_text_description = obj_text_description.to(self.device) # bs
                # now concatenate the "waterbird" and "landbird" text vectors to text_description and also create it's corresponding labels
                texts = torch.cat((self.tokenized_text_inputs, obj_text_description), dim=0).to(self.device) # bs+2
                gt_label_text = torch.cat((gt_og_input_text, gt_label_batch), dim=0).to(self.device) # bs+2
                logits_text = self.model.encode_text(texts) # bs+2 * 1024
                
                # concatenate the image and text features and their label vectors too
                logits = torch.cat((logits_image, logits_text), dim=0) # bs + bs+2 * 1024
                labels_image_text = torch.cat((gt_label_batch, gt_label_text), dim=0) # bs + bs+2
                loss = loss_total(logits, labels_image_text)
            

            elif self.config.mode == 'yu_yang_mitigating':
                images = images.to(self.device) # bs
                logits_image = self.model.encode_image(images) # bs * 1024'

                obj_text_description = obj_text_description.to(self.device) # bs
                # now concatenate the "waterbird" and "landbird" text vectors to text_description and also create it's corresponding labels
                texts = torch.cat((self.tokenized_text_inputs, obj_text_description), dim=0).to(self.device) # bs+2
                gt_label_text = torch.cat((gt_og_input_text, gt_label_batch), dim=0).to(self.device) # bs+2
                logits_text = self.model.encode_text(texts) # bs+2 * 1024

                #################
                # Exact same as above case until here. But then we apply the function differently
                #################
                pos_weight = 1
                neg_weight = 1
                sep_pos_neg = False
                abs = True
                loss = similarity_loss(logits_image, logits_text, gt_label_batch, gt_label_text, pos_weight, neg_weight, sep_pos_neg, abs)

            
            elif self.config.mode == 'separate_text_image':

                ###################
                # Image case
                ###################
                # image case is same for both models whether we consider background or not so writing that outside the background if

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
                    logits_text = self.model.encode_text(texts) # bs+2 * 1024
                    logits_text = logits_text.unsqueeze(0) # 1 * (bs+2) * 1024
                    logits_text = logits_text.repeat(images.shape[0], 1, 1) # bs * (bs + 2) * 1024

                # In this case we don't
                else:
                    logits_text = self.model.encode_text(obj_text_description).unsqueeze(0) # 1 * bs * 1024
                    logits_text = logits_text.repeat(logits_text.shape[1], 1, 1) # bs * bs* 1024
                    gt_label_text = gt_label_batch.unsqueeze(0) #  1 * bs
                    gt_label_text = gt_label_text.repeat(gt_label_text.shape[1], 1, 1)

                # concatenate the image and text features and their label vectors too to get 16*19*1024 or 16*17*1024
                logits = torch.cat((logits_image, logits_text), dim=1) # bs* bs+3 * 1024 or bs* bs+1 * 1024
                # print(logits_image.size())
                # print(logits_text.size())
                # print(logits.size())
                labels_image_text = torch.cat((gt_label_image, gt_label_text), dim=1) # bs* bs+3 or bs* bs+1
                # print(labels_image_text.size())
                image_loss_term = loss_img(logits, labels_image_text)

                
                ###################
                # Text case
                ###################

                # in this case use only the object text description
                if self.config.background_consider == False:
                    obj_text_description = obj_text_description.to(self.device) # bs
                    # now concatenate the "waterbird" and "landbird" text vectors to text_description and also create it's corresponding labels
                    texts = torch.cat((self.tokenized_text_inputs, obj_text_description), dim=0).to(self.device) # bs+2
                    gt_label_text = torch.cat((gt_og_input_text, gt_label_batch), dim=0).to(self.device) # bs+2
                    logits_text = self.model.encode_text(texts) # bs+2 * 1024
                    text_loss_term = loss_text(logits_text, gt_label_text)

                # In this case the background is considered but the background text is negative to both the text classes
                else:
                    obj_text_description = obj_text_description.to(self.device) # bs
                    background_text_description = background_text_description.to(self.device) # bs
                    texts = torch.cat((self.tokenized_text_inputs, obj_text_description, background_text_description), dim=0).to(self.device) # 2bs+2
                    gt_label_text = torch.cat((gt_og_input_text, gt_label_batch_text_desc, gt_label_batch_background), dim=0).to(self.device) # 2bs+2
                    logits_text = self.model.encode_text(texts) # 2bs+2 * 1024
                    text_loss_term = loss_text(logits_text, gt_label_text)

                loss = image_loss_term + text_loss_term

            # One thing to note: In this loss case we are not matching waterbird class to waterbird text at all supervised. 
            # For example:
            # We are just matching all waterbirds on water images to other waterbirds on water images. And we are matching the text descriptions of all waterbirds on water images to text descriptions from other waterbirds on water images.
            elif self.config.mode == 'separate_text_image_yes_bg_4classes':
                # add background classes to classnames to create 4 different strings of classes
                new_classes = np.array(self.text_classes)[:, None] + np.array(self.bg_classes) # converting to numpy column vector to do broadcasting
                class_to_index = {cls: idx for idx, cls in enumerate(new_classes.flatten())} # gives 4 classes

                class_names_of_sample = np.char.add(gt_label_image_text, gt_label_background) # bs # this does elementwise string addition so if it's waterbird and water then waterbirdwater
                gt_label_batch = torch.tensor([class_to_index[label] for label in class_names_of_sample], device=self.device).unsqueeze(1) # bs

                #########################################################
                # REST OF THIS HAS TO BE FILLED IN
                ##########################################################


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.append(loss.item())
            loss_dict = {'loss': train_loss[-1], 'iteration': self.current_iteration}
            self.logger.add_scalar(f'TrainLoss', loss_dict['loss'], loss_dict['iteration'])

            per_sample_loss = train_loss[-1] / self.config.batch_size
            print(f'Iteration {iteration} done with per closs {per_sample_loss}.')

            if iteration % test_iteration == 0 or iteration == total_iterations:
                self.test_dict['test_acc']['iter_no'].append(self.current_iteration)

                # saves the model according to test frequency
                print("Saving model before testing")
                self.save_model(self.model, self.optimizer, best=False)
                self.test_during_train() # saves the model again if it's best here

                self.model = self.weight_mode(self.model, trainable=True, probe=self.config.probe, network_type=self.config.network_type)
                self.model.train()
                self.convert_models_to_fp32(self.model)

            self.current_iteration += 1

            del sampled_batch, images, texts
            torch.cuda.empty_cache()

        return

    def test_during_train(self):

        self.test_dict['csv'] = {'Image_Name': [], 'GT': [], f'pred{self.current_iteration:04d}': []}
        img_name, gt_labels, scores = clip_inference(self.model, self.tokenized_text_inputs, self.test_loader, self.device)
        
        self.test_dict['csv'][f'pred{self.current_iteration:04}'] = scores
        self.test_dict['csv']['Image_Name'] = img_name
        self.test_dict['csv']['GT'] = gt_labels

        accuracy, class_accuracy = calculate_class_pred_accuracy(self.text_classes, gt_labels, scores)

        self.test_dict['csv'][f'pred{self.current_iteration:04}'].append(accuracy)
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
    scratch_dir = os.getenv("SCRATCH")
    results_dir = scratch_dir + '/results/Train'

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--test_epochs', type=int, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=8e-5)
    parser.add_argument('--num_gpu_workers', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default=results_dir)
    parser.add_argument('--model_type', type=str, default='ViT-B/32', choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'SGD'])
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--base_temperature', type=float, default=0.07)
    parser.add_argument('--mode', type=str, default='separate_text_image', choices=['separate_text_image', 'text_image_concat_no_bg', 'yu_yang_mitigating', 'separate_text_image_yes_bg_4classes'])
    parser.add_argument('--background_consider', type=bool, default=False)
    parser.add_argument('--include_classtext_in_image_training', type=bool, default=False)
    parser.add_argument('--llava_out_dir', type=str, default=f'{scratch_dir}/pipeline_results0000')

    parser.add_argument('--network_type', type=str, default= 'clip', choices=['clip', 'modified_clip'])
    parser.add_argument('--layer_type_for_modified_clip', type=str, default= 'linear', choices=['mlp', 'linear'])
    parser.add_argument('--probe', type=bool, default=False)

    config = parser.parse_args()
    return config


def train_clip_align_simultaneously(run_dir, mode, learning_rate, include_classtext_in_image_training = False, epochs = 10, background_consider=False, network_type='clip', layer_type_for_modified_clip='linear', probe=False):
    config = configuration_params()
    config.llava_out_dir = run_dir
    config.results_dir = run_dir
    config.mode = mode
    config.learning_rate = learning_rate
    config.include_classtext_in_image_training = include_classtext_in_image_training
    config.epochs = epochs
    config.test_epochs = (epochs * 1.0) / 10.0
    config.background_consider = background_consider
    config.network_type = network_type
    config.layer_type_for_modified_clip = layer_type_for_modified_clip
    config.probe = probe
    scratch_dir = os.getenv("SCRATCH")
    img_dir = scratch_dir + '/datasets'
    model = Align_CLIP(model_type=config.model_type, dataset='waterbirds', text_classes=['waterbird', 'landbird'], bg_classes= ['water', 'land'], img_dir=img_dir, config=config)
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