
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
            module.back_res = F.linear(input[0], module.weight, module.bias)
        
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

def compare_act_diff(module,name):
    """
    通过插入forward的前hook和后hook，来计算用不用sefp的计算结果的区别
    """

    def pre_forward_hook(module, input):
        module.back_res = module(input[0])
        
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
    
    