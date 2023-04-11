import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from Res2_DTEN import Res2_DTEN
from datasets import test_dataset
import cv2



        
        
        
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./model/20000.pth')

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = './TestDataset/{}/images/'.format(_data_name)
   
    save_path = './results/vit_originaldetr/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = Res2_DTEM()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    
    image_root = data_path
    gt_root = '/home/guangyu/csp/projects/PraNet/data/TestDataset/gt/{}/'.format(_data_name)
   
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    
    
    

    for i in range(test_loader.size):
        image, gt, name,mask = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        mask = torch.from_numpy(mask).cuda().unsqueeze(0)
       
       
        res = model(image,mask)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = torch.sigmoid(res).detach().cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res *= 255.0
        res = res.astype(np.uint8)
       
        cv2.imwrite(save_path+name, res)
        
        
