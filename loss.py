from matplotlib import pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage import filters
from sklearn.preprocessing import normalize
import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import torch.nn as nn

from config import get_config
config = get_config()

def image_gradient(image):
    edges_x = filters.sobel_h(image)
    edges_y = filters.sobel_v(image)
    edges_y = normalize(edges_y)
    edges_x = normalize(edges_x)
    return torch.from_numpy(edges_x), torch.from_numpy(edges_y)


def image_gradient_loss(image, pred):
    loss = 0
    for i in range(len(image)):
        pred_grad_x, pred_grad_y = image_gradient(pred[i][0].cpu().detach().numpy())
        gray_image = torch.from_numpy(rgb2gray(image[i].permute(1, 2, 0).cpu().numpy()))
        image_grad_x, image_grad_y = image_gradient(gray_image)
        IMx = torch.mul(image_grad_x, pred_grad_x).float()
        IMy = torch.mul(image_grad_y, pred_grad_y).float()
        Mmag = torch.sqrt(torch.add(torch.pow(pred_grad_x, 2), torch.pow(pred_grad_y, 2))).float()
        IM = torch.add(torch.ones(config.image_size, config.image_size), torch.neg(torch.pow(torch.add(IMx, IMy), 2)))
        numerator = torch.sum(torch.mul(Mmag, IM))
        denominator = torch.sum(Mmag)
        loss = loss + torch.div(numerator, denominator)
    return torch.div(loss, len(image))


class HairMatLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(HairMatLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.loss = 0
        self.num_classes = config.num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, image, mask, gradient_loss_ratio):
        print(pred.shape)
        print(image.shape)
        print(mask.shape)
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        
        """
        from matplotlib import pyplot as plt
        plt.subplot(1,3,1)
        plt.imshow(pred[1,:,:,0].cpu().detach().numpy())
        plt.subplot(1,3,2)
        plt.imshow(mask[1,0,:,:].long().cpu().detach().numpy())
        plt.subplot(1,3,3)
        plt.imshow(image[1,:,:,:].cpu().detach().numpy().transpose(1, 2, 0))
        plt.show()
        """
        # has to permute here for data order in pytorch is B*C*H*C
        pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
        mask_flat = mask.squeeze(1).view(-1).long()
        
        cross_entropy_loss = F.cross_entropy(pred_flat, mask_flat, weight=self.weight
                                             , ignore_index=self.ignore_index, reduction=self.reduction)
        if gradient_loss_ratio > 0:
            image_loss = image_gradient_loss(image, pred).to(self.device)
            return torch.add((1-gradient_loss_ratio)*cross_entropy_loss, gradient_loss_ratio*image_loss.float())
        else:
            return cross_entropy_loss

class HairMatSoftmaxLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(HairMatSoftmaxLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.loss = 0
        self.num_classes = config.num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, image, mask, gradient_loss_ratio):
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        """
        from matplotlib import pyplot as plt
        plt.subplot(1,3,1)
        plt.imshow(pred[1,:,:,0].cpu().detach().numpy())
        plt.subplot(1,3,2)
        plt.imshow(mask[1,0,:,:].long().cpu().detach().numpy())
        plt.subplot(1,3,3)
        plt.imshow(image[1,:,:,:].cpu().detach().numpy().transpose(1, 2, 0))
        plt.show()
        """
        # does not have to permute here for I permuted in the model already 
        # to support converting the model to ncnn format for ncnn

        pred_flat = pred.contiguous().view(-1, self.num_classes)
        mask_flat = mask.squeeze(1).view(-1).long()
        
        cross_entropy_loss = F.cross_entropy(pred_flat, mask_flat, weight=self.weight
                                             , ignore_index=self.ignore_index, reduction=self.reduction)
        if gradient_loss_ratio > 0:
            image_loss = image_gradient_loss(image, pred).to(self.device)
            return torch.add((1-gradient_loss_ratio)*cross_entropy_loss, gradient_loss_ratio*image_loss.float())
        else:
            return cross_entropy_loss

def iou_softmax_loss(pred, mask):
    pred = pred.permute(0, 3, 1, 2)
    pred = torch.argmax(pred, 1).long()
    mask = torch.squeeze(mask).long()
    
    # print("pred:", pred)
    # print("mask:", mask)
    # something wrong here
    # print(mask.shape)
    # plt.imshow(mask.to('cpu').numpy()[0,:,:], cmap=plt.cm.gray)
    # plt.show()
    # plt.imshow(pred.to('cpu').numpy()[0,:,:], cmap=plt.cm.gray)
    # plt.show()
    
    Union = torch.where(pred > mask, pred, mask)
    Overlep = torch.mul(pred, mask)
    loss = torch.div(torch.sum(Overlep).float(), torch.sum(Union).float())
    return loss

def iou_loss(pred, mask):
    
    """ 
    print(mask.shape)
    print(pred.shape)
    plt.subplot(1,2,1)
    plt.imshow(mask.long().to('cpu').detach().numpy()[0,0,:,:], cmap=plt.cm.gray)
    plt.subplot(1,2,2)
    plt.imshow(pred.to('cpu').detach().numpy()[0,0,:,:], cmap=plt.cm.gray)
    plt.show()
    """

    Union = torch.where(pred > mask, pred, mask)
    Overlep = torch.mul(pred, mask)
    loss = torch.div(torch.sum(Overlep).float(), torch.sum(Union).float())
    return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
    def forward(self, inputs, targets):
        targets = targets.long()
        return self.nll_loss(F.log_softmax(inputs), targets)


def dice_loss(input, target):
    
    
    smooth = 1e-5
    iflat = input.view(-1)
    tflat = target.view(-1)

    intersection = (iflat*tflat).sum()
        
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

 
def fusion_softmax_loss(input, target, img):
    """
    fusion loss where the softmax dim is in the last channel, which
    means in binary porblem the last channel is 2
    """
    """
    from matplotlib import pyplot as plt
    pred = input
    mask = target
    plt.subplot(1,2,1)
    plt.imshow(pred[1,:,:,0].cpu().detach().numpy())
    plt.subplot(1,2,2)
    plt.imshow(mask[1,0,:,:].long().cpu().detach().numpy())
    # plt.subplot(1,3,3)
    # plt.imshow(image[1,:,:,:].cpu().detach().numpy().transpose(1, 2, 0))
    plt.show()
    """
    """ cross entropy loss
    num_classes = 2
    mask = target
    mask[mask < 0.5] = 0
    mask[mask >=0.5] = 1
    
    pred_flat = input.contiguous().view(-1, num_classes)
    mask_flat = mask.squeeze(1).view(-1).long()
    criterion = nn.CrossEntropyLoss()
    cross_entropy_loss = criterion(pred_flat, mask_flat)
    """

    alpha_input_flat = input[:,:,:,1].contiguous().view(-1)
    alpha_target_flat = target.squeeze(1).view(-1)
    alpha_loss = torch.sqrt(torch.pow(alpha_input_flat - alpha_target_flat, 2.)+1e-6).mean()
    
    gt_img = torch.cat((target, target, target), 1) * img
    alpha = input.permute(0, 3, 1, 2)[:,1:,:,:]
    alpha_img = torch.cat((alpha, alpha, alpha), 1) * img
    color_loss = torch.sqrt(torch.pow(gt_img - alpha_img, 2.) + 1e-6).mean()

    return alpha_loss +  color_loss

   
def fusion_loss(alpha, target, img):

    alpha_loss = torch.sqrt(torch.pow(alpha - target, 2.) + 1e-6).mean()

    gt_img = torch.cat((target, target, target), 1) * img
    alpha_img = torch.cat((alpha, alpha, alpha), 1) * img
    color_loss = torch.sqrt(torch.pow(gt_img - alpha_img, 2.) + 1e-6).mean()

    return alpha_loss + color_loss
