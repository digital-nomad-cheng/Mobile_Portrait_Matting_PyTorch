import os
import argparse

import cv2
import numpy as np
from scipy.misc import imread, imresize
from PIL import Image
import torch
import torchvision

from config import get_config
config = get_config()
if config.model_name == 'PrismaNet':
    from nets.PrismaNet import PrismaNet
elif config.model_name == 'PrismaMattingNet':
    from nets.PrismaMattingNet import PrismaNet

def  blend_img_with_mask(img, alpha, result_path, img_name):
    img = np.array(img)
    mask = alpha >= 0.99
    mask_n = np.zeros(img.shape, dtype='float32')
    mask_n[:,:,0] = 255
    mask_n[:,:,0] *= alpha
    result = img*0.5 + mask_n*0.5
    result = np.clip(np.uint8(result), 0, 255)
    Image.fromarray(result).save(os.path.join(result_path, config.model_name+'_'+img_name+'_blend.jpg'))

def evaluate(data_path, model_path, result_path):
    model = PrismaNet()
    model.load_state_dict(torch.load(model_path))
    imgs = [f for f in os.listdir(data_path) if not f.startswith('.')]
    for _ in imgs:
        img_name = _.split('.')[0]
        img = Image.open(os.path.join(data_path, _)).resize((config.image_size, config.image_size))
        inputs = np.array(img)
        inputs = inputs.transpose(2, 0, 1).astype('float32')
        inputs = (inputs - 127.5)/127.5
        inputs = np.expand_dims(inputs, axis=0)
        input_tensor = torch.tensor(inputs)
        with torch.set_grad_enabled(False):
            model.eval()
            output_tensor = model(input_tensor)
            if 'Matting' in config.model_name:
                foreground = output_tensor[0,0,:,:].detach().numpy()
            else:
                foreground = output_tensor[0,:,:,1].detach().numpy()
            Image.fromarray((foreground*255).astype('uint8')).save(os.path.join(result_path, config.model_name+'_'+img_name+'_alpha.jpg'))
            blend_img_with_mask(img, foreground, result_path, img_name)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, default='./imgs/test/')
    # parser.add_argument('--model_path', type=str, default='./models/PrismaNet_portrait_epoch-0099.pth')
    # args = parser.parse_argument()
    evaluate(config.test_data_path, config.test_model_path, config.test_result_path)


