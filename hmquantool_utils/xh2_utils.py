
import torch.nn as nn
import torch
import torch.nn.functional as F

import types
from copy import copy, deepcopy
from hmquant.ptq.sefp.hm_sefp import conv_hm_fp_quant_forward

from ..data_process_utils.data_compare_utils import \
    Sensitive2Function, SENSITIVES, print_diff_info

TableDiff = {}

def compare_linear_diff(module,name):
    """
    通过插入forward的前hook和后hook，来计算用不用sefp的计算结果的区别
    """

    def pre_forward_hook(module, input):
        if module.weight is not None:
            if module.bias is not None:
                module.back_res = F.linear(input[0], module.weight.to(input[0]), module.bias.to(input[0]))
            else:
                module.back_res = F.linear(input[0], module.weight.to(input[0]), None)
        
    def post_forward_hook(module, input, output):
        if module.back_res is not None:
            back_output = module.back_res
            module.back_res = None
            
            diff_dict = {}
            for sens in SENSITIVES:
                diff_dict[sens] = Sensitive2Function[sens](back_output,output)
            # print(name)
            # print_diff_info(diff_dict)
            TableDiff[name[14:]] = diff_dict
    
    module.register_forward_pre_hook(pre_forward_hook)
    module.register_forward_hook(post_forward_hook)

def compare_rmsnorm_diff(module,name):
    """
    通过插入forward的前hook和后hook，来计算用不用sefp的计算结果的区别
    """

    def pre_forward_hook(module, input):
        rms = torch.sqrt(input[0].float().pow(2).mean(-1, keepdim=True) + module.variance_epsilon).to(module.weight)
        normalized = input[0] / rms
        module.back_res = module.weight * normalized
        
    def post_forward_hook(module, input, output):
        if module.back_res is not None:
            back_output = module.back_res
            module.back_res = None
            
            diff_dict = {}
            for sens in SENSITIVES:
                diff_dict[sens] = Sensitive2Function[sens](back_output,output)
            # print(name)
            # print_diff_info(diff_dict)
            TableDiff[name[14:]] = diff_dict
    
    module.register_forward_pre_hook(pre_forward_hook)
    module.register_forward_hook(post_forward_hook) 
    
def compare_layernorm_diff(module,name):
    """
    通过插入forward的前hook和后hook，来计算用不用sefp的计算结果的区别
    """

    def pre_forward_hook(module, input):
        module.back_res = F.layer_norm(
            input[0], module.normalized_shape, module.weight, 
            module.bias, module.eps)
        if module.back_res.isnan().any():
            print(name,"has nan torch output")
        
    def post_forward_hook(module, input, output):
        if input[0].isnan().any():
            print(name,"has nan input")
        if output.isnan().any():
            print(name,"has nan output")
        if module.back_res is not None:
            back_output = module.back_res
            module.back_res = None
            
            diff_dict = {}
            for sens in SENSITIVES:
                diff_dict[sens] = Sensitive2Function[sens](back_output,output)
            # print(name)
            # print_diff_info(diff_dict)
            TableDiff[name] = diff_dict
    
    module.register_forward_pre_hook(pre_forward_hook)
    module.register_forward_hook(post_forward_hook) 

def find_inf_nan(module,name):    
    def post_forward_hook(module, input, output):
        for i in range(len(input)):
            if input[i].isnan().any():
                print(name,f" {i}'s has nan input")
            if input[i].isinf().any():
                print(name,f" {i}'s has inf input")
                
        for i in range(len(output)):      
            if output[i].isnan().any():
                print(name,f" {i}'s has nan output")
            if output[i].isinf().any():
                print(name,f" {i}'s has inf output")
            
    module.register_forward_hook(post_forward_hook) 
 