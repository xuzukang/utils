import torch
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
import argparse
import torch.nn as nn

import functools
from tqdm import tqdm
torch.set_grad_enabled(False)
# import pdb

import itertools

import os
import sys
ROOT = os.getcwd()
sys.path.append(str(ROOT)+"/vim_quant")



def get_matmul_act_scales(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}
    from quantize.int_matmul import QuantMatMul
    def stat_tensor(name, tensor):
        tmp = tensor.permute(0,1,3,2)
        hidden_dim = tmp.shape[-1]
        tmp = tmp.reshape(-1, hidden_dim).abs().detach()
        # comming_max = torch.max(tmp, dim=0)[0].float().cpu()
        comming_max = torch.quantile(tmp, 0.999999, dim=0).float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, QuantMatMul):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    subset_dataloader = itertools.islice(dataloader, num_samples)
    for batch in tqdm(subset_dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
        if isinstance(batch, list):
            images, target = batch
        else:
            images, target = batch["image"], batch["label"]
        model(images.to(device))

    for h in hooks:
        h.remove()

    return act_scales

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2', help='model name')
    parser.add_argument("--resume", type=str, default='saved_checkpoint/vim_t_midclstok_76p1acc.pth')
    parser.add_argument("--batch_size", type=int, default=10, help="batch size.")
    parser.add_argument('--scales-output-path', type=str, default='./act_scales/',help='where to save the act scales')
    parser.add_argument('--shifts-output-path', type=str, default='./act_shifts/',help='where to save the act shifts')
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",choices=["wikitext2", "ptb", "c4", "mix","pile"],help="Where to extract calibration data from.",)
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    args = parser.parse_args()
    return args


@torch.no_grad()
def vim_generate_matmul_scale():
    from timm.models import create_model
    import vim_quant.vim.models_mamba
    from vim_quant.vim.datasets import build_dataset
    args = parse_args()
    
    # resum_path = "vim_quant/saved_checkpoint/vim_t_midclstok_76p1acc.pth"
    # model_name = "vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
    # net_name = "vim-tiny"
    resum_path = "vim_quant/saved_checkpoint/vim_s_midclstok_80p5acc.pth"
    model_name = "vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
    net_name = "vim-small"
    output_path = "vim_quant/saved_checkpoint"
    batch_size = 1
    num_samples = 128
    
    
    device = torch.device('cuda')

    lm  = create_model(
        model_name,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224
    )
    lm.to(device)
    lm.eval()

    checkpoint = torch.load(resum_path, map_location='cpu')
    lm.load_state_dict(checkpoint['model'])
    from vim_quant.vim.utils import convert_vim_2_vim_torch
    convert_vim_2_vim_torch(lm,device)

    args.data_set = 'IMNET'
    args.data_path = "/data01/datasets/imagenet"
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val=torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(batch_size),
        num_workers=batch_size*2,
        pin_memory=True,
        drop_last=False
    )
    
    model = lm
    
    #ptq
    from vim.normalized_modules import MatMul
    from quantize import QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d
    w_cfg = {"dynamic_method":"per_tensor","n_bits":8}
    a_cfg = {"dynamic_method":"per_tensor","n_bits":8}
    def replace_layers(model, target_class, replacement_class):
        for name, child in model.named_children():
            if isinstance(child, target_class):
                # Replace the layer with the new quantized version
                if target_class == MatMul:
                    setattr(model, name, replacement_class(x1_quant_params=w_cfg,x2_quant_params=a_cfg))
                else:
                    setattr(model, name, replacement_class(child,weight_quant_params=w_cfg,act_quant_params=a_cfg))
            else:
                # Recursively call this function on the child module
                replace_layers(child, target_class, replacement_class)

    # Usage example:
    # Assuming QuantMatMul, QuantLinear, QuantConv1d, QuantConv2d are defined
    replace_layers(model, MatMul, QuantMatMul)
    replace_layers(model, nn.Linear, QuantLinear)
    replace_layers(model, nn.Conv1d, QuantConv1d)
    replace_layers(model, nn.Conv2d, QuantConv2d)
    from quantize.utils import set_quant_state
    set_quant_state(model,weight_quant=True,act_quant=True)

    from vim.hm_model_utils import fuse_layer_norms, RotateModule, RQuantLinear
    from vim.hadamard import random_hadamard_matrix
    h1 = random_hadamard_matrix(model.layers[0].mixer.in_proj.in_features,device)
    R1 = RotateModule(h1)
    h2 = random_hadamard_matrix(model.layers[0].mixer.out_proj.in_features,device)
    R2 = RotateModule(h2)
    model.register_parameter("R1",R1.weight)
    
    h3 = random_hadamard_matrix(model.layers[0].mixer.out_proj.in_features,device)
    R3 = RotateModule(h3)

    h4 = random_hadamard_matrix(16,device)
    R4 = RotateModule(h4)
    
    if True:
        
        fuse_layer_norms(model)
        def substitute_layers(model):
            for name,module in model.named_modules():
                if 'in_proj' in name and 'quantizer' not in name:
                    new_module = RQuantLinear(module,R1=R1,transpose=False)
                elif 'out_proj' in name and 'quantizer' not in name:
                    new_module = RQuantLinear(module,R1=R1,R2=R2,transpose=True)
                elif 'matmul' in name and 'quantizer' not in name:
                    # module.register_parameter("R3",R3.weight)
                    # module.register_parameter("R4",R4.weight)
                    continue
                else:
                    continue
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                if parent_name:  
                    parent = dict(model.named_modules())[parent_name]
                    setattr(parent, name.split('.')[-1], new_module)
                else:  
                    setattr(model, name, new_module)

        substitute_layers(model)

    set_quant_state(model,weight_quant=False,act_quant=False)
    
    
    act_scales = get_matmul_act_scales(model, data_loader_val,num_samples)
    save_path = os.path.join(output_path,f'{net_name}_matmul_scale.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)


def mamba2d_classify_generate_matmul_scale():
    from mmengine.config import Config, ConfigDict, DictAction
    from mmengine.runner import Runner
    from model_image_classification.src.mamba import Mamba2DModel
    from model_image_classification.utils.datasets import build_dataset
    args = parse_args()
    model_cfg = './model_image_classification/config/mamba2d_b.py'
    cfg = Config.fromfile(model_cfg)
    cfg.model_ckpt= "./ckpt/mamba2d_b.pth"
    cfg.work_dir = './work_dirs/mamba2d'
    runner = Runner.from_cfg(cfg)
    
    runner._test_loop = runner.build_test_loop(runner._test_loop)  # type: ignore

    runner.call_hook('before_run')

    # make sure checkpoint-related hooks are triggered after `before_run`
    runner.load_or_resume()
    runner.hooks[1]._swap_ema_parameters()
    
    from model_image_classification.utils.utils import convert_vim_2_vim_torch
    convert_vim_2_vim_torch(runner.model.backbone,"cuda")
    output_path = "model_image_classification/ckpt/"
    net_name = cfg.model_ckpt.split("/")[-1].split(".")[0]
    batch_size = 1
    num_samples = 128
    
    
    device = torch.device('cuda')

    args.data_set = 'IMNET'
    args.data_path = "/data01/datasets/imagenet"
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val=torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(batch_size),
        num_workers=batch_size*2,
        pin_memory=True,
        drop_last=False
    )
    


    w_cfg = {"dynamic_method":"per_tensor","n_bits":8}
    a_cfg = {"dynamic_method":"per_tensor","n_bits":8}
    from model_image_classification.utils.normalized_modules import MatMul
    from quantize import QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d
    from quantize.utils import set_quant_state
    def replace_layers(model, target_class, replacement_class):
        for name, child in model.named_children():
            # if 'matmul' in name:
            #     continue
            if isinstance(child, target_class):
                # Replace the layer with the new quantized version
                if target_class == MatMul:
                    setattr(model, name, replacement_class(x1_quant_params=w_cfg,x2_quant_params=a_cfg))
                else:
                    setattr(model, name, replacement_class(child,weight_quant_params=w_cfg,act_quant_params=a_cfg))
            else:
                # Recursively call this function on the child module
                replace_layers(child, target_class, replacement_class)
    
    replace_layers(runner.model, MatMul, QuantMatMul)
    set_quant_state(runner.model,weight_quant=False,act_quant=False)
    
    act_scales = get_matmul_act_scales(runner.model, data_loader_val,num_samples)
    save_path = os.path.join(output_path,f'{net_name}_matmul_scale.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)



if __name__ == '__main__':
    # vim_generate_matmul_scale()
    mamba2d_classify_generate_matmul_scale()
