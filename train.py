import argparse
from datasets import *
from experiments import *
from utils import *
from tqdm import tqdm
import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'

def parse_opt():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='GSS')
  parser.add_argument('--dataset', type=str, default='')
  parser.add_argument('--source_dir', type=str, default='')
  parser.add_argument('--target_dir', type=str, default='')
  parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
  parser.add_argument('--output_dir', type=str, default='./outputs')
  parser.add_argument('--num_iters', type=int, default = 5000)
  parser.add_argument('--test_interval', type=int, default = 250)
  parser.add_argument('--save_interval', type=int, default = 1000)
  parser.add_argument('--batch_size', type=int, default = 16)
  parser.add_argument('--img_size', type=int, default=256)
  parser.add_argument('--crop_size', type=int, default=256)
  parser.add_argument('--mode', type=str, default='train_prior') #train_posterior --> train_prior
  opt, _ = parser.parse_known_args()
  return opt

def main():
  torch.manual_seed(0)
  opt = parse_opt()
  model, config = load_model(opt)
  model = model.cuda()
  build_files(opt, config)

  if os.path.exists(config['ckpt_dir_post']):
    print('-- load weights successfully -- ')
    model.load_weights()

  src_loader, tar_loader, test_loader = build_dataset(opt, config)
  loop = tqdm(range(opt.num_iters))
  for i in loop:
    if (i + 1) % opt.test_interval == 0:
      model.M.eval()
      model.I.eval()
      if opt.mode == 'train_posterior':
        viz_maskige(src_loader, model, i, config)
      elif opt.mode == 'train_prior':
        viz_seg(src_loader, tar_loader, model, i, config)
      if (i + 1) % opt.save_interval == 0:
        model.save_weights()
    if i % len(src_loader) == 0:
      iter_src_loader = iter(src_loader)
    if i % len(tar_loader) == 0:
      iter_tar_loader = iter(tar_loader)
    model.M.train()
    model.I.train()
    x_s, y_s = next(iter_src_loader)
    x_t, y_t = next(iter_tar_loader)
    x_s, x_t, y_s, y_t = x_s.cuda(), x_t.cuda(), y_s.cuda(), y_t.cuda()

    if opt.mode == 'train_posterior':
      l_cls = model.train_posterior(x_s, x_t, y_s, y_t, i)
    elif opt.mode == 'train_prior':
      l_cls = model.train_prior(x_s, x_t, y_s, y_t, i)
    loop.set_postfix(
      loss = f'--l_cls: {l_cls}')
  model.save_hist()

if __name__ == '__main__':
  main()

