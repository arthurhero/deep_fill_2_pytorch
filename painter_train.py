'''
Training code for painter GAN
'''
import os
import sys
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import utils.opencv_utils as cv
import utils.tools as tl
import utils.dataloader as dl

import painter_ops as po

bg_training_flist_file='dataset/MITindoor/Images/training.txt'
bg_validation_flist_file='dataset/MITindoor/Images/validation.txt'
bg_testing_flist_file='dataset/MITindoor/Images/testing.txt'
bg_gen_ckpt_path='logs/bg_gen_128_pad_rep_freeze.ckpt'
bg_gen_coarse_ckpt_path='logs/bg_coarse_gen_128_pad_rep_freeze.ckpt'
bg_dis_ckpt_path='logs/bg_dis_128_pad_rep_freeze.ckpt'

bg_in_channels=3
gan_iteration=5
batch_size=16
img_size=256
epoch=100
lr=0.001
l1_alpha=1
l1_coarse_alpha=1
fm_alpha=0
patch_alpha=0.05

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ImageDataSet(Dataset):
    '''
    Images dataset
    (MITindoor and Rendered 3D Models from ShapeNet)
    with free-form mask
    '''
    def __init__(self, flist_file, alpha):
        self.flist=dl.load_flist_file(flist_file)
        self.alpha=alpha

    def __len__(self):
        return len(self.flist)

    def __getitem__(self,idx):
        img = cv.load_img(self.flist[idx])
        if img is None:
            sys.exit("Error! Img is None at: "+self.flist[idx])
        img = dl.process_img(img,crop_size=img_size,resize=False,sample_num=1,alpha=self.alpha,normalize=True,pytorch=True,random_mask=True,ones_boundary=False)
        img=img[0]
        img=torch.from_numpy(img)
        return img

def train_painter(pretrain=False,fix_coarse=False,ob=False):
    '''
    pretrain - whether this is training the coarse part of generator or not
    ob - whether this is training an object painter instead of a bg painter
    '''
    in_channels=3
    gen_ckpt_path=bg_gen_ckpt_path
    gen_coarse_ckpt_path=bg_gen_coarse_ckpt_path
    dis_ckpt_path=bg_dis_ckpt_path
    training_flist_file=bg_training_flist_file
    if ob:
        in_channels=4
        gen_ckpt_path=ob_gen_ckpt_path
        gen_coarse_ckpt_path=ob_gen_coarse_ckpt_path
        dis_ckpt_path=ob_dis_ckpt_path
        training_flist_file=ob_training_flist_file
    #load dataset
    alpha=False
    if ob:
        alpha=True
    dataset=ImageDataSet(training_flist_file,alpha=alpha)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    gennet=po.PainterNet(in_channels,pretrain,fix_coarse,device).to(device)
    gennet.train()
    disnet=po.SNPatchGAN(in_channels,device).to(device)
    disnet.train()
    gen_optimizer = torch.optim.Adam(gennet.parameters(),lr=lr,betas=(0.5,0.9))
    dis_optimizer = torch.optim.Adam(disnet.parameters(),lr=lr,betas=(0.5,0.9))

    #load checkpoint
    if pretrain:
        if os.path.isfile(gen_coarse_ckpt_path):
            gennet.load_state_dict(torch.load(gen_coarse_ckpt_path))
            print("Loaded coarse gen ckpt!")
    else:
        if os.path.isfile(gen_ckpt_path):
            gennet.load_state_dict(torch.load(gen_ckpt_path))
            print("Loaded gen ckpt!")
        else:
            if os.path.isfile(gen_coarse_ckpt_path):
                gennet.load_state_dict(torch.load(gen_coarse_ckpt_path))
                print("Loaded coarse gen ckpt! Training fine from coarse now!")
    if os.path.isfile(dis_ckpt_path):
        disnet.load_state_dict(torch.load(dis_ckpt_path))
        print("Loaded dis ckpt!")

    for e in range(epoch):
        step=0
        for i,img_batch in enumerate(dataloader):
            train_g=False
            if pretrain or step%(gan_iteration+1)==gan_iteration:
                #if False:
                train_g=True
                disnet.apply(tl.freeze_params)
                gennet.apply(tl.unfreeze_params)
            else:
                disnet.apply(tl.unfreeze_params)
                gennet.apply(tl.freeze_params)
            actual_batch_size=img_batch.shape[0]
            img_batch=img_batch.to(device)
            imgs=img_batch[:,:in_channels]
            masks=img_batch[:,in_channels:]
            incomplete_imgs=imgs*(masks.eq(0.).float())

            #get predictions from generator
            predictions=None
            x_coarse=None
            if pretrain:
                x_coarse=gennet(incomplete_imgs,masks)
                predictions=x_coarse
            else:
                x_coarse,x=gennet(incomplete_imgs,masks)
                predictions=x*masks+incomplete_imgs

            #get score from discriminator
            pos_neg_in=torch.cat([imgs,predictions],dim=0)
            pos_neg_score,pos_neg_feature=disnet(pos_neg_in,masks)
            pos_score=pos_neg_score[:actual_batch_size]
            neg_score=pos_neg_score[actual_batch_size:]
            pos_feature=pos_neg_feature[:actual_batch_size]
            neg_feature=pos_neg_feature[actual_batch_size:]

            #calculate losses
            if not train_g:
                d_loss_pos=F.relu(torch.ones_like(pos_score)-pos_score)
                d_loss_neg=F.relu(torch.ones_like(neg_score)+neg_score)
                d_loss=(d_loss_pos).mean()+(d_loss_neg).mean()
                dis_optimizer.zero_grad()
                d_loss.backward()
                dis_optimizer.step()
                if step%(gan_iteration+1)==0:
                    print('Epoch [{}/{}] , Step {}, D_Loss: {:.4f}, Pos_avg_score: {:.4f}, Neg_avg_score: {:.4f}'
                            .format(e+1, epoch, step, d_loss.item(), pos_score.mean().item(), neg_score.mean().item()))
            else:
                l1_loss=(predictions-imgs).abs().mean()
                feature_match_loss=(pos_feature-neg_feature).abs().mean()
                loss=l1_loss*l1_alpha+feature_match_loss*fm_alpha
                if not fix_coarse:
                    l1_coarse_loss=(x_coarse-imgs).abs().mean()
                    loss+=l1_coarse_loss*l1_coarse_alpha
                if not pretrain:
                    g_loss=-(neg_score).mean()
                    loss+=(g_loss*patch_alpha)
                gen_optimizer.zero_grad()
                loss.backward()
                gen_optimizer.step()

                if pretrain:
                    torch.save(gennet.state_dict(), gen_coarse_ckpt_path)
                else:
                    torch.save(gennet.state_dict(), gen_ckpt_path)
                torch.save(disnet.state_dict(), dis_ckpt_path)
                print('Epoch [{}/{}] , Step {}, G_Loss: {:.4f}, l1_loss: {:.4f}, g_loss: {:.4f}'
                        .format(e+1, epoch, step, loss.item(), l1_loss.item(), g_loss.item()))

            step+=1

train_painter()
