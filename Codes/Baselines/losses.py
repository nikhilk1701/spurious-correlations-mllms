# Code to: Organize all of the loss functions for the model in one place
# Date Created: 10/24/2024
# Last Modified By: Shika

import torch
import torch.nn as nn
import traceback
import datetime
import time


# This is from this paper: https://arxiv.org/pdf/2004.11362
# Github code taken from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py (official)
# This takes in [bsz, n_views, feature_dim] for feature vector. Changing that to take just [bsz, feature_dim]
# So modified the loss from github
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, contrast_mode='all'):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None):
        """
        Args:
            features: hidden vector of shape [bsz, feature_dim].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device) # [16, 16]

        # compute logits
        if self.contrast_mode == 'one':
            anchor_feature = features[0, :].unsqueeze(0) # taking the first feature vector as anchor [1, 1024]
            mask = mask[0, :].unsqueeze(0) # taking the first mask [1, 16]
        elif self.contrast_mode == 'all':
            anchor_feature = features
        
        contrast_feature = features # all feature vectors are contrast features [16, 1024]

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # taking along the last dimension
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size).view(-1, 1).to(device),0)
        mask = mask * logits_mask 

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


# Just rewrote the above loss only with torch.bmm instead of torch.matmul. This way it can handle no_of_images, no_of_features, feature_dim [16, 17, 1024]
# Could just do the above loss 16 times for each image in a loop then mean but that would be inefficient
class BatchedSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, contrast_mode='all'):
        super(BatchedSupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_features, feature_dim].
            labels: ground truth of shape [bsz, n_features].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        num_features = features.shape[1]
        labels_expanded = labels.unsqueeze(1) # [16, 1, 17]	
        mask = (labels_expanded == labels_expanded.transpose(1, 2)).float().to(device) # [16, 17, 17]

        # compute logits
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0, :].unsqueeze(0) # taking the first feature vector as anchor [16, 1, 1024]
            mask = mask[:, 0, :].unsqueeze(0) # taking the first mask [16, 1, 17]
        elif self.contrast_mode == 'all':
            anchor_feature = features
        
        contrast_feature = features # all feature vectors are contrast features [16, 17, 1024]

        anchor_dot_contrast = torch.div(torch.bmm(anchor_feature, contrast_feature.transpose(1, 2)), self.temperature) # [16, 17, 17] if all
        logits_max, _ = torch.max(anchor_dot_contrast, dim=2, keepdim=True) # taking along the last dimension
        logits = anchor_dot_contrast - logits_max.detach() # [16, 17, 17]

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(num_features).view(-1, 1).to(device),0) # [17, 17]
        logits_mask = logits_mask.unsqueeze(0).repeat(batch_size, 1, 1) # [16, 17, 17]
        mask = mask * logits_mask # [16, 17, 17]

         # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask # [16, 17, 17]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # [16, 17, 17]

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1) 
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

# Directly copying the code from Yu Yang mitigating paper
def similarity_loss(features_1, features_2, labels_1, labels_2, pos_weight, neg_weight, sep_pos_neg=False, abs=True):
    """
    Args:
        features_1: vector of shape [bsz, feature_dim]. (could be text or image)
        features_2: vector of shape [bsz, feature_dim]. (could be text or image)
        labels_1: ground truth of shape [bsz].
        labels_2: ground truth of shape [bsz].
        sep_pos_neg: If True, return pos_loss and neg_loss separately.
        abs: If True, use the absolute value of the negative loss.
    Returns:
        A loss scalar.
    """
    sim_fun = nn.CosineSimilarity(dim=2, eps=1e-6)
    loss = sim_fun(features_1.unsqueeze(1), features_2.unsqueeze(0))

    pos_mask = (labels_1.unsqueeze(1).expand(-1, len(labels_2))-labels_2.unsqueeze(0).expand(len(labels_1),-1)) == 0
    pos_mask = pos_mask.float().cuda()

    pos_loss = (loss * pos_mask).mean(dim=-1)
    neg_loss = (loss * (1- pos_mask)).mean(dim=-1)
    if abs:
        neg_loss = neg_loss.abs()

    if sep_pos_neg:
        return (pos_loss, neg_loss)
    else:
        return - pos_weight * pos_loss + neg_weight * neg_loss

    
#################################################
# TESTING OUR LOSSES 
#################################################
    
def test_supcon_loss():
    loss1 = SupervisedContrastiveLoss(contrast_mode= 'all')
    loss2 = SupervisedContrastiveLoss(contrast_mode= 'one')

    features = torch.randn(4, 1024)
    labels = torch.randint(0, 3, (4,)) # 3 classes

    output1 = loss1(features, labels)
    output2 = loss2(features, labels)
    
    print("Output for contrast_mode = all: ", output1)
    print("Output for contrast_mode = one: ", output2)
    return 

def test_batched_supcon_loss():
    loss1 = BatchedSupervisedContrastiveLoss(contrast_mode= 'all')
    loss2 = BatchedSupervisedContrastiveLoss(contrast_mode= 'one')

    features = torch.randn(4, 5, 1024)
    labels = torch.randint(0, 3, (4, 5)) # 3 classes

    output1 = loss1(features, labels)
    output2 = loss2(features, labels)
    
    print("Output for contrast_mode = all: ", output1)
    print("Output for contrast_mode = one: ", output2)
    return

# Don't have to test the other loss cause directly copied from the paper but just in case we want to
def test_similarity_loss():
    features_1 = torch.randn(4, 1024)
    features_2 = torch.randn(4, 1024)
    labels_1 = torch.randint(0, 3, (4,)) # 3 classes
    labels_2 = torch.randint(0, 3, (4,)) # 3 classes

    pos_weight = 1
    neg_weight = 1
    sep_pos_neg = False
    abs = True

    output = similarity_loss(features_1, features_2, labels_1, labels_2, pos_weight, neg_weight, sep_pos_neg, abs)
    print("Output for similarity loss: ", output)
    return

if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:

        test_supcon_loss()
        # test_batched_supcon_loss()
        # test_similarity_loss()

        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))