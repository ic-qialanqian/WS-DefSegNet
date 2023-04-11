import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import joint_transforms
from config import train_data
from datasets_fully import ImageFolder
from misc import AvgMeter, check_mkdir
from Res2_DTEN import Res2_DTEN
from torch.backends import cudnn
import torch.nn.functional as functional
import numpy as np
import cv2
import torch.nn.functional as F


cudnn.benchmark = True

torch.manual_seed(2018)
torch.cuda.set_device(0)




##########################hyperparameters###############################
ckpt_path = './model'
exp_name = 'res2net_dten'
args = {
    'iter_num':20000,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 1e-4,
    'lr_decay': 0.9,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'snapshot': ''
}
##########################data augmentation###############################
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(352,352),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
target_transform = transforms.ToTensor()
##########################################################################
train_set = ImageFolder(train_data, joint_transform, img_transform, target_transform,352)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)
criterion = nn.BCEWithLogitsLoss().cuda()
criterion_BCE = nn.BCELoss().cuda()
criterion_MAE = nn.L1Loss().cuda()
criterion_MSE = nn.MSELoss().cuda()

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')




    


def main():
    model = Res2_DTEN()
    net = model.cuda().train()
    #net.load_state_dict(torch.load("./model/vit_polyp_lr1e-5/10000.pth"))
    
    
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])





    if len(args['snapshot']) > 0:
        print ('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

        #optimizer.param_groups[0]['lr'] = args['lr']
        #optimizer.param_groups[1]['lr'] = 10*args['lr']


    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)

def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record,loss3_record,loss4_record,loss5_record,loss6_record,loss7_record,loss8_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 *args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']
            inputs, masks, gts,padding_mask= data
            gts[gts>0.5] = 1
            gts[gts!=1] = 0

            masks[masks>0.5] = 1
            masks[masks!=1] = 0

            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            masks = Variable(masks).cuda()
            gts = Variable(gts).cuda()
            padding_mask = Variable(padding_mask).cuda()
            
            padding_mask = padding_mask.squeeze(1)
            
            
            outputs,dten_out = net(inputs,padding_mask)
            
            ##########loss#############
            optimizer.zero_grad()
            
            
            
            
            loss1 = structure_loss(outputs,gts)

            
            
            loss2 = structure_loss(dten_out,gts)
            
            total_loss = loss1 + loss2

            
            total_loss.backward()
            
            optimizer.step()
            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            loss2_record.update(loss2.item(), batch_size)
            
            curr_iter += 1
            #############log###############
            if curr_iter %10000==0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                
            log = '[iter %d], [total loss %.5f],[loss_res %.5f],[loss_dten %.5f],[lr %.13f]'  % \
                     (curr_iter, total_loss_record.avg, loss1_record.avg,loss2_record.avg,optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')
            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                
                return
            #############end###############

if __name__ == '__main__':
    main()
