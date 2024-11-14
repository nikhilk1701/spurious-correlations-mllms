# Code to: Finetune CLIP by adding some layers to it
# Date Created: 11/13/2024
# Last Modified By: Shika

import torch
import torch.nn as nn
import torch.nn.init as init

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
        elif self.layer_type == "mlp":
            self.projection_head = nn.Sequential(
                nn.Linear(self.clip_model.visual.output_dim, self.clip_model.visual.output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.clip_model.visual.output_dim, self.clip_model.visual.output_dim)
            )
        
        # Apply Xavier initialization to all Linear layers in the projection_head
        if isinstance(self.projection_head, nn.Linear):
            xavier_init(self.projection_head)
        elif isinstance(self.projection_head, nn.Sequential):
            for module in self.projection_head:
                if isinstance(module, nn.Linear):
                    xavier_init(module)
    
    def forward(self, x):
    
        clip_image_features = self.clip_model.encode_image(x)
        clip_image_features = self.projection_head(clip_image_features)
        
        return clip_image_features
    

class TextCLIPModified(nn.Module):
    def __init__(self, clip_model, layer_type= "linear"):  
        super(TextCLIPModified, self).__init__()
        self.clip_model = clip_model
        self.layer_type = layer_type

        if self.layer_type == "linear":
            self.projection_head = nn.Linear(self.clip_model.transformer.shape[2], self.clip_model.transformer.shape[2])
        elif self.layer_type == "mlp":
            self.projection_head = nn.Sequential(
                nn.Linear(self.clip_model.transformer.shape[2], self.clip_model.transformer.shape[2]),
                nn.ReLU(inplace=True),
                nn.Linear(self.clip_model.transformer.shape[2], self.clip_model.transformer.shape[2])
            )
        elif self.layer_type == "throwaway":
            self.projection_head = nn.Sequential(
                nn.Linear(self.clip_model.transformer.shape[2], self.clip_model.transformer.shape[2]),
                nn.ReLU(inplace=True),
                nn.Linear(self.clip_model.transformer.shape[2], 128)
            )
        
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
            clip_text_features = self.projection_head(clip_text_features)
            return clip_text_features
        else:
            if throw_it:
                clip_text_features = self.clip_model.encode_text(x)
            else:
                clip_text_features = self.projection_head(self.clip_model.encode_text(x))
            return clip_text_features
    

class CLIPCombinedModified(nn.Module):
    def __init__(self, model, layer_type= "linear"):
        super(ImageCLIPModified, self).__init__()
        self.layer_type = layer_type

        self.model = model

        if self.layer_type == "linear":
            self.visual = ImageCLIPModified(self.model, self.layer_type)
            self.text_model = TextCLIPModified(self.model, self.layer_type)
        elif self.layer_type == "mlp":
            self.visual = ImageCLIPModified(self.model, self.layer_type)
            self.text_model = TextCLIPModified(self.model, self.layer_type)
        elif self.layer_type == "throwaway":
            self.visual = ImageCLIPModified(self.model, self.layer_type)
            self.text_model = TextCLIPModified(self.model, self.layer_type)

    # Setting the defaults for inference as True as don't want to change the eval function files also
    def encode_image(self, x):
        return self.visual(x)
    
    def encode_text(self, x, throw_it=True):
        return self.text_model(x, throw_it)
