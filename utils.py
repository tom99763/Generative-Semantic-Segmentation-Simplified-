import yaml
from models import GSS
import os

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def load_model(opt):
    config = get_config(f'./configs/GSS.yaml')
    model = GSS.Trainer(config, opt)
    return model, config

def build_files(opt, config):
    file_name = f"{opt.model}_{config['params']['num_classes']}_{config['params']['num_embs']}"
    src = opt.source_dir.split('/')[-1]
    tar = opt.target_dir.split('/')[-1]
    output_dir = f'{opt.output_dir}/{opt.dataset}'
    ckpt_dir = f'{opt.ckpt_dir}/{opt.dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    output_dir = f'{output_dir}/{src}2{tar}'
    ckpt_dir = f'{ckpt_dir}/{src}2{tar}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    output_dir = f'{output_dir}/{file_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ckpt_dir = f'{ckpt_dir}/{file_name}'
    for name in ['viz_post', 'viz_prior', 'stats']: #viz_post, viz_prior
        if not os.path.exists(f'{output_dir}/{name}'):
            os.makedirs(f'{output_dir}/{name}')
    config['ckpt_dir_post'] = f'{ckpt_dir}_post.pt'
    config['ckpt_dir_prior'] = f'{ckpt_dir}_prior.pt'
    config['output_dir'] = output_dir
