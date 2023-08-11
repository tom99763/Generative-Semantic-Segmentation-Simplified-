import sys
sys.path.append('./models')
from vqvae import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class MaskigeTT(nn.Module):
    def __init__(self, config):
        super().__init__()
        #params
        self.num_classes = config['params']['num_classes']

        #vqvae
        self.vqvae = Dalle_VAE()
        for param in self.vqvae.parameters():
            param.requires_grad = False

        #linear map
        self.mask2rgb = nn.Sequential(
            nn.Conv2d(self.num_classes, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1)
        )

        #non linear map
        self.rgb2mask = nn.Sequential(
            nn.Conv2d(3, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, self.num_classes, 1)
        )

    def forward_maskige(self, gt):
        gt = gt.float()

        #map to pixel
        segmap = self.mask2rgb(gt).sigmoid()

        #encode
        feat_probs = self.vqvae.get_codebook_probs(segmap)

        #decode
        rec_segmap = self.vqvae.forward(feat_probs)[:, :3].sigmoid()

        #map to mask
        seg_logits = self.rgb2mask(rec_segmap) #(b, k, h, w)
        return seg_logits

    def forward_maskige_test(self, gt):
        gt = gt.float()

        # map to pixel
        segmap = self.mask2rgb(gt).sigmoid()

        # encode
        feat_idx = self.vqvae.get_codebook_indices(segmap)

        # decode
        rec_segmap = self.vqvae.decode(feat_idx)[:, :3].sigmoid()
        seg_logits = self.rgb2mask(rec_segmap)
        return seg_logits, segmap, rec_segmap


    def get_feat_idx(self, gt):
        gt = gt.float()
        segmap = self.mask2rgb(gt).sigmoid()
        # encode
        feat_idx = self.vqvae.get_codebook_indices(segmap)
        return feat_idx

    def forward(self, x):
        x = x.argmax(dim=1)
        seg_map = self.vqvae.decode(x)[:, :3].sigmoid()
        logits = self.rgb2mask(seg_map)
        return logits, seg_map

backbones = {'resnet18': models.resnet18,
             'resnet34': models.resnet34,
             'resnet50': models.resnet50,
             'resnet101': models.resnet101
            }

class GenerativeSegHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone_name = config['network']['backbone']
        resnet = backbones[backbone_name](weights='DEFAULT')
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.proj = nn.Conv2d(960, 8192, 1)
    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        h, w = e2.shape[-2:]
        resize = transforms.Resize((h, w), antialias=True)
        e = torch.cat([resize(e1), e2, resize(e3), resize(e4)], dim=1)
        f = self.proj(e)
        return f
