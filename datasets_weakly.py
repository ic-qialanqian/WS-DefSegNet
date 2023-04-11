import os
import os.path
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])[0]
        mask = self.rgb_loader(self.images[self.index])[1]
        image = self.transform(image).unsqueeze(0)
        #mask = self.gt_transform(mask)#.unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        #gt = self.gt_transform(gt)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name,mask

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img_height, img_width = img.size
            if img.size[0] or img.size[1]>= self.testsize:
                # padding_mask = np.zeros((352,352))
                padding_mask = np.bool_(np.zeros([self.testsize,self.testsize]))
            else:
                paved_image,padding_mask = random_pave(img, [self.testsize,self.testsize], limit=16)
            return img.convert('RGB'),padding_mask

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')




def make_dataset(root):
    img_path = os.path.join(root, 'images')
    depth_path = os.path.join(root, 'scribble_masks_750')
    gt_path = os.path.join(root, 'scribble_gts_750')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_path) if f.endswith('.png')]
    
    return [(os.path.join(img_path, img_name + '.png'),
             os.path.join(depth_path,img_name + '.png'),os.path.join(gt_path, img_name + '.png')) for img_name in img_list]


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None,trainsize=None):
        self.root = root
        self.trainsize = trainsize
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img_path, depth_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        depth = Image.open(depth_path).convert('L')
        if img.size[0] or img.size[1]>= self.trainsize:
                # padding_mask = np.zeros((352,352))
            padding_mask = np.bool_(np.zeros([self.trainsize,self.trainsize]))
        else:
            paved_image,padding_mask = random_pave(img, [self.trainsize,self.trainsize], limit=16)

        padding_mask = Image.fromarray(padding_mask)
        if self.joint_transform is not None:
            img, depth, target,padding_mask = self.joint_transform(img,depth, target,padding_mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            depth = self.target_transform(depth)
            padding_mask = self.target_transform(padding_mask)

        return img, depth,target,padding_mask

    def __len__(self):
        return len(self.imgs)
