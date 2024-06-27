import torch
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import datetime
import glob
import os
from typing import Callable, Any, Optional, Union, List,Dict,Tuple
from torch.utils.data import IterableDataset, DataLoader
from hmquant.ptq.sequencer_module import Sequencer, switch_mode_temporary,SequencerNode,Node
from hmquant.IR import GraphABC, NodeABC, PASSIVE_OPERATIONS, GraphNodeABC
from hmquant.tools.parser.onnx2seq.onnx_utils import get_layer_sensitivities,\
                                                    table_info,\
                                                    Sensitive2Function,\
                                                    SENSITIVES
from hmquant.tools.parser.onnx2seq.onnx_utils import BaseAnalyzer,\
                                                get_sequencer_results,\
                                                get_layer_sensitivities,\
                                                table_info
from hmquant.logger.logger import logger
from copy import deepcopy

def save_sequencer_2_pkl(sequencer, path,name):
    sequencer.save_pkl(path, name)

def load_pkl_model(path):
    # 打开文件并使用 pickle.load() 加载对象
    import pickle
    from hmquant.api import Sequencer
    with open(path, 'rb') as file:
        qmodel = pickle.load(file)
    return qmodel

def map_aggregate(fn, in_put):
    if isinstance(in_put, list):
        return [map_aggregate(fn, i) for i in in_put]
    elif isinstance(in_put, tuple):
        return tuple(map_aggregate(fn, i) for i in in_put)
    elif isinstance(in_put, dict):
        return {k: map_aggregate(fn, v) for k, v in in_put.items()}
    return fn(in_put)

def quant_some_act_weght_sequencer_forward(sequencer:Sequencer,
                                           inputs, 
                                           quant_scopes=[0,99999],
                                           not_quant_layers=[]):
    current_node_num=-1

    # 1. 准备输入
    if not isinstance(inputs[0], dict):
        graph_input_ids = sequencer.graph_input_ids
        inputs = {input_id: inputs[i] for i, input_id in enumerate(graph_input_ids)}
    else:
        inputs = inputs[0]
    def fn_process(t):
        _t = t.to(sequencer.device) if isinstance(t, torch.Tensor) else t
        if hasattr(t, "quant_param"):
            _t.quant_param = t.quant_param
        return _t
    inputs = map_aggregate(fn_process, inputs)
    # Reset
    out_counts = deepcopy(sequencer.out_counts)
    sequencer.results = {}
    # 2. 根据节点执行
    outputs_dict = dict()
    for node in  sequencer.nodes:
        node_id = node.id
        if node_id in not_quant_layers:node.not_quant_flag=True
        if node.op_type == "Input":
            out = inputs.pop(node_id)
        else:
            ins = map_aggregate(lambda i: sequencer.results[i], node.input_ids)
            if hasattr(node, "not_quant_flag") and node.not_quant_flag:
                current_node_num += 1
                print('not quant layer:   ',node_id)
                node.op.mode = "raw"
                out = node.op(*ins)
            elif isinstance(node, GraphNodeABC):
                node.set_ops_mode('quant_forward')
                out = node.forward(*ins)
            else:
                current_node_num += 1
                if quant_scopes[0]<=current_node_num<quant_scopes[1]:
                    node.op.mode='quant_forward' 
                    out = node.op(*ins)
                    # print(current_node_num,'    ', node.id)
                else:
                    # node.op.mode='quant_forward' 
                    # out = node.op(*ins)
                    node.op.mode='raw'
                    out_raw = node.op(*ins)
                    # out.data = out_raw.data
                    out=out_raw
                # print("output:  ",node.id, '    ', out.shape)
                # print()
            print(current_node_num, '   ', node_id) #注意，这里是从1开始计数，且这个current_node_num是不被包含在量化层里面的，也就是如果判断出那一层出现的问题，这一层应该减2对应层号才是正真的层号
        if node.op_type == "Output":
            outputs_dict[node_id] = out
        else:
            sequencer.results[node_id] = out
        if isinstance(out, torch.Tensor):
            sequencer.results_shape[node_id] = out.shape
            sequencer.results_type[node_id] = out.dtype
        for id in set(node.input_ids):
                out_counts[id].remove(node_id)
                if len(out_counts[id]) == 0:
                    if id not in sequencer.cache_id:
                        del sequencer.results[id]
                    else:
                        sequencer.results[id] = map_aggregate(
                            lambda t: t.detach().cpu()
                            if isinstance(t, torch.Tensor)
                            else t,
                            sequencer.results[id],)
    return outputs_dict


def get_layer_sensitivities_from_diff_sequencer_with_id(raw_results:Dict[Node,Union[torch.Tensor,Tuple]], 
                                                quant_results:Dict[Node,Union[torch.Tensor,Tuple]], 
                                                sequencer, num_samples, layer_sensitivities):
        for node in raw_results.keys():
            qual_name = node# qual_name = node.qualified_name
            raw_out = raw_results[node]
            for node_quant in quant_results.keys():
                if qual_name == node_quant:
                    quant_out = quant_results[node_quant]
                    break
            if qual_name not in layer_sensitivities:
                layer_sensitivities[qual_name] = {sens:0 for sens in SENSITIVES}
                layer_sensitivities[qual_name]['op_type'] = qual_name.split('_')[0]
            if not isinstance(raw_out,torch.Tensor):
                continue
            for sens in SENSITIVES:
                if raw_out.dtype == torch.bool:
                    raw_out = raw_out.to(torch.float32)
                    quant_out = quant_out.to(torch.float32)
                layer_sensitivities[qual_name][sens] += (
                    Sensitive2Function[sens](
                        raw_out.cpu().detach().numpy(),
                        quant_out.cpu().detach().numpy(),
                    )
                    / num_samples
                )

def analyse_quant_some_act_and_weight_function(sequencer:Sequencer,
                                               focus_layes:bool,
                                               calib_dataset:list,
                                               save_path,
                                               scope=[0,1000]):
    sequencer_copy = deepcopy(sequencer)

    not_quant_layers = []

    calib_dataset = calib_dataset[41:101:3]
    index = list(range(1,340,1)) 
    res_to_polt = {}
    for i in index:#总长638
        quant_scopes = [-1,i]
    # if scope:
    #     quant_scopes = scope#左闭右开    
    #     i = scope[-1]
        

        # for operation in sequencer_copy.get_operations(True,False):
        #     if hasattr(operation.op, "w_observer") and (operation.op.w_observer is not None):
        #         operation.op.weight = torch.nn.Parameter(operation.op.w_quant_param.quant_tensor(operation.op.weight.data))
        
        sequencer.set_device('cuda')
        sequencer.set_cache_id("all")
        sequencer_copy.set_device('cuda')
        sequencer_copy.set_cache_id("all")
        
        res_to_polt[str(i)]={}
        for focus_layer in focus_layes:
            res_to_polt[str(i)][focus_layer] = {}
       
        for seq_index, seq_in in enumerate(calib_dataset):
            if isinstance(seq_in, dict):
                seq_inputs = [
                    seq_in[name]
                    if isinstance(seq_in[name], torch.Tensor)
                    else torch.from_numpy(seq_in[name])
                    for name in sequencer.graph_input_ids
                ]
            elif isinstance(seq_in, torch.Tensor):
                seq_inputs = [seq_in]
            else:
                raise NotImplementedError()

            sequencer.set_ops_mode("raw")
            if seq_index == -1:
                analy = BaseAnalyzer('cpu')
                analy.hook_net(sequencer)
            sequencer.forward(*seq_inputs, get_output_dict=True)
            raw_results = sequencer.results#get_sequencer_results(sequencer)
            
            sequencer_copy.set_ops_mode("raw")
            quant_some_act_weght_sequencer_forward(sequencer_copy,seq_inputs,
                                                    quant_scopes=quant_scopes,
                                                    not_quant_layers=not_quant_layers)
            quant_results = sequencer_copy.results

            if seq_index == -1:
                analy.remove_hooks()

            
            from hmquant.tools.parser.onnx2seq.onnx_utils import table_info
            layer_sensitivities = dict()
            get_layer_sensitivities_from_diff_sequencer_with_id(raw_results, quant_results, sequencer, 1, layer_sensitivities)
            for focus_layer in focus_layes:
                for sens in SENSITIVES:
                    if layer_sensitivities.get(focus_layer):
                        res_to_polt[str(i)][focus_layer][sens] = layer_sensitivities[focus_layer][sens]

        for focus_layer in focus_layes:
            for sens in SENSITIVES:
                if layer_sensitivities.get(focus_layer):
                    res_to_polt[str(i)][focus_layer][sens] /= len(calib_dataset)
                    logger.info(f"{sens} of {focus_layer}: {res_to_polt[str(i)][focus_layer][sens]}")
        table_info(layer_sensitivities, "量化误差分析表，按余弦相似度从低到高排序", use_order=False)
        sequencer.clear_cache()
        sequencer_copy.clear_cache()


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for focus_layer in focus_layes:
        for sens in SENSITIVES:
            data = [res_to_polt[str(i)][focus_layer][sens] for i in index]
            fig, ax = plt.subplots(figsize=(48, 6))
            ax.plot(index,data)
            ax.set_xticks(index)  # 设置刻度位置
            ax.set_xticklabels(index, rotation=45)
            ax.tick_params(axis='x', labelsize=7)
            ax.vlines(index, ymin=0, ymax=data, colors='gray', linestyles='--')
            plt.savefig(save_path+focus_layer.replace('/', '-') + "-" +sens +".png")
            plt.close()

      