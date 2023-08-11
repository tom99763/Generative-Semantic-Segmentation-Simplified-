from dall_e import load_model
from math import sqrt
import os
import sys
sys.path.append('../QuantizedSeg')
import torch
import torch.nn.functional as F

class BasicVAE(nn.Module):
    def get_codebook_indices(self, images):
        raise NotImplementedError()

    def decode(self, img_seq):
        raise NotImplementedError()

    def get_codebook_probs(self, img_seq):
        raise NotImplementedError()

    def get_image_tokens_size(self):
        pass

    def get_image_size(self):
        pass


class Dalle_VAE(BasicVAE):
    def __init__(self):
        super().__init__()
        self.encoder = load_model( "./checkpoints/encoder.pkl", 'cuda')
        self.decoder = load_model( "./checkpoints/decoder.pkl", 'cuda')

    def get_codebook_indices(self, images):
        z_logits = self.encoder(images)
        return torch.argmax(z_logits, axis=1)

    def get_codebook_probs(self, images):
        z_logits = self.encoder(images)
        return nn.Softmax(dim=1)(z_logits)

    def decode(self, img_seq):
        z = F.one_hot(img_seq, num_classes=8192).permute(0, 3, 1, 2).float()
        return self.decoder(z).float()

    def forward(self, img_seq_prob, img_size=None, no_process=True):
        if no_process:
            return self.decoder(img_seq_prob.float()).float()
        else:
            bs, seq_len, num_class = img_seq_prob.size()
            z = img_seq_prob.view(bs, img_size, img_size, 8192)
            return self.decoder(z.permute(0, 3, 1, 2).float()).float()
