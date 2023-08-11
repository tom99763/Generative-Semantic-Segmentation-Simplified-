from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import os.path
from pre_process import *
import torch
import torch.nn.functional as F

def build_dataset(opt, config):
    num_classes = config['params']['num_classes']
    transform_img_train, transform_label_train = train_transform(opt.img_size, opt.img_size)
    transform_img_test, transform_label_test = train_transform((448, 224), 448, aug=False)
    #transform_test = test_transform(opt.img_size, opt.crop_size, opt.backbone == 'AlexNet')
    src_list = ImageList(make_list(opt.source_dir), num_classes, transform_img_train, transform_label_train)
    tar_list = ImageList(make_list(opt.target_dir), num_classes, transform_img_test, transform_label_test)
    test_list = ImageList(make_list(opt.target_dir), num_classes, transform_img_test, transform_label_test)
    src_loader = DataLoader(src_list, batch_size = opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
    tar_loader = DataLoader(tar_list, batch_size = opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_list, batch_size = 36, shuffle=False, num_workers=4, drop_last=False)
    return src_loader, tar_loader, test_loader

def make_list(dir_):
    list_ = sorted(os.listdir(f'{dir_}/images'))
    img_list = sorted(list(map(lambda img_name: f'{dir_}/images/{img_name}', list_)))
    mask_list = sorted(list(map(lambda img_name: f'{dir_}/masks/{img_name}', list_)))
    img_list = tuple(zip(img_list, mask_list))
    return img_list

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, img_list, num_classes, transform_img=None, transform_mask=None, mode='RGB'):
        self.imgs = img_list
        self.num_classes = num_classes
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.rgb_loader = rgb_loader
        self.gray_loader = l_loader
    def __getitem__(self, idx):
        path, mask_path = self.imgs[idx]
        img = self.rgb_loader(path)
        mask = self.gray_loader(mask_path)
        img = self.transform_img(img)
        mask = self.transform_mask(mask) #(c, h, w)
        mask = torch.where(mask > 0., 1., 0.).long()
        mask = F.one_hot(mask[0], self.num_classes)
        mask = mask.permute(2, 0, 1)
        return img, mask
    def __len__(self):
        return len(self.imgs)

