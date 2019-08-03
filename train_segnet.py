import os
from glob import glob

import torch
import matplotlib.pyplot as plt
import numpy as np

from nets.segnet import SegNet
from loss import dice_loss, CrossEntropyLoss2d, iou_softmax_loss, fusion_softmax_loss


class Trainer:
    def __init__(self, config, dataloader):
        self.batch_size = config.batch_size
        self.config = config
        self.lr = config.lr
        self.epoch = config.epoch
        self.checkpoint_dir = config.checkpoint_dir
        self.model_path = config.checkpoint_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = dataloader
        self.image_len = len(dataloader)
        self.num_classes = config.num_classes
        self.build_model()
        self.sample_step = config.sample_step
        self.sample_dir = config.sample_dir
    
    def build_model(self):
        self.net = SegNet()
        self.net.to(self.device)
        self.load_model()

    def load_model(self):
        print(" * Load checkpoint in ", str(self.model_path))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.listdir(self.model_path):
            print(" ! No checkpoint in ", str(self.model_path))
            return

        model = glob(os.path.join(self.model_path, "SegNet_portrait_*.pth"))
        model.sort()
        
        
        if len(model) != 0:
            self.net.load_state_dict(torch.load(model[-1], map_location=self.device))
            print(" * Load Model from %s: " % str(self.model_path), str(model[-1]))

    def train(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-7)

        for epoch in range(self.epoch):
            for step, (image, mask) in enumerate(self.data_loader):
                image = image.to(self.device)
                mask = mask.to(self.device)
                pred = self.net(image)
                self.net.zero_grad()
                loss = fusion_softmax_loss(pred, mask, image)

                loss.backward()
                optimizer.step()

                iou = iou_softmax_loss(pred, mask)
                 
                print("epoch: [%d/%d] | image: [%d/%d] | loss: %.4f | IOU: %.4f" % (epoch, self.epoch, step, self.image_len, loss, iou))
                
                # save sample images
                if step % self.sample_step == 0:
                    self.save_sample_imgs(image[0], mask[0], pred[0], self.sample_dir, epoch, step)
                    
                    print('[*] Saved sample images')

            torch.save(self.net.state_dict(), '%s/SegNet_portrait_epoch-%04d.pth' % (self.checkpoint_dir, epoch))

    def save_sample_imgs(self, real_img, real_mask, prediction, save_dir, epoch, step):
        data = [real_img, real_mask, prediction[:,:,1]]
        names = ["Image", "Mask", "Prediction"]
        
        fig = plt.figure()
        for i, d in enumerate(data):
            d = d.squeeze()
            im = d.data.cpu().numpy()

            if i == 1 or i == 2:
                im = np.expand_dims(im, axis=0)
                im = np.concatenate((im, im, im), axis=0)
            
            im = im.transpose(1,2,0).astype('float32') 
            f = fig.add_subplot(1, 3, i + 1)
            f.imshow(im)
            f.set_title(names[i])
            f.set_xticks([])
            f.set_yticks([])

        p = os.path.join(save_dir, "epoch-%s_step-%s.png" % (epoch, step))
        plt.savefig(p)
