# Code to: Finetune CLIP by adding some layers to it
# Date Created: 11/13/2024
# Last Modified By: Shika

import torch
import torch.nn as nn
import torch.nn.init as init

import numpy as np

# Throwaway doesn't make sense to me in train simultaneously case as the text_image_matching is not decoupled
# Like if we train the 2 throwaway layers at the same time, it won't make sense
# But in separate training case it does?
# First train text encoder with a contrastive head and throw it away at inference
# Then train image encoder without a throwaway head but with the text encoder after throwing away the head

# Function to apply Xavier initialization
def xavier_init(layer):
    if isinstance(layer, nn.Linear):
        init.xavier_uniform_(layer.weight)  # Xavier uniform initialization for weights
        if layer.bias is not None:
            init.zeros_(layer.bias)  # Initialize bias to zero

class ImageCLIPModified(nn.Module):
    def __init__(self, clip_model, layer_type= "linear"):
        super(ImageCLIPModified, self).__init__()
        self.clip_model = clip_model
        self.layer_type = layer_type

        if self.layer_type == "linear":
            self.projection_head = nn.Linear(self.clip_model.visual.output_dim, self.clip_model.visual.output_dim)
        if self.layer_type == "mlp":
            self.projection_head = nn.Sequential(
                nn.Linear(self.clip_model.visual.output_dim, self.clip_model.visual.output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.clip_model.visual.output_dim, self.clip_model.visual.output_dim)
            )

        # if there is a projection head do this
        if self.layer_type != "none":
            # Apply Xavier initialization to all Linear layers in the projection_head
            if isinstance(self.projection_head, nn.Linear):
                xavier_init(self.projection_head)
            elif isinstance(self.projection_head, nn.Sequential):
                for module in self.projection_head:
                    if isinstance(module, nn.Linear):
                        xavier_init(module)
    
    def forward(self, x):
        clip_image_features = self.clip_model.encode_image(x)
        if self.layer_type == "none":
            return clip_image_features
        else:
            clip_image_features = self.projection_head(clip_image_features)
            return clip_image_features
    

class TextCLIPModified(nn.Module):
    def __init__(self, clip_model, layer_type= "linear"):  
        super(TextCLIPModified, self).__init__()
        self.clip_model = clip_model
        self.layer_type = layer_type

        enc_dim = self.clip_model.transformer.width

        if self.layer_type == "linear":
            self.projection_head = nn.Linear(enc_dim, enc_dim)
        elif self.layer_type == "mlp":
            self.projection_head = nn.Sequential(
                nn.Linear(enc_dim, enc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(enc_dim, enc_dim)
            )
        elif self.layer_type == "throwaway":
            self.projection_head = nn.Sequential(
                nn.Linear(enc_dim, enc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(enc_dim, 128)
            )
        
        # Apply Xavier initialization to all Linear layers in the projection_head
        # if there is a projection head do this
        if self.layer_type != "none":
            # Apply Xavier initialization to all Linear layers in the projection_head
            if isinstance(self.projection_head, nn.Linear):
                xavier_init(self.projection_head)
            elif isinstance(self.projection_head, nn.Sequential):
                for module in self.projection_head:
                    if isinstance(module, nn.Linear):
                        xavier_init(module)
    
    def forward(self, x, throw_it):
        if self.layer_type != "throwaway":
            clip_text_features = self.clip_model.encode_text(x)
            if self.layer_type == "none":
                return clip_text_features
            else:
                clip_text_features = self.projection_head(clip_text_features)
                return clip_text_features
        else:
            if throw_it:
                clip_text_features = self.clip_model.encode_text(x)
            else:
                clip_text_features = self.projection_head(self.clip_model.encode_text(x))
            return clip_text_features
    

class CLIPCombinedModified(nn.Module):
    def __init__(self, model, layer_type_image= "linear", layer_type_text= "linear"):
        super(CLIPCombinedModified, self).__init__()
        self.layer_type_image = layer_type_image
        self.layer_type_text = layer_type_text

        self.model = model

        self.visual = ImageCLIPModified(self.model, self.layer_type_image)
        self.text_model = TextCLIPModified(self.model, self.layer_type_text)

    # Setting the defaults for inference as True as don't want to change the eval function files also
    def encode_image(self, x):
        return self.visual(x)
    
    def encode_text(self, x, throw_it=True):
        return self.text_model(x, throw_it)
    
    def forward(self, img, text):
        image_features = self.encode_image(img)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

