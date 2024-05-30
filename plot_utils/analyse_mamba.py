import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from regex import P
import torch
import torch.nn as nn
import random
from datasets import load_dataset
from transformers import AutoTokenizer, MambaForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from utils.fake_quant_utils.fake_quant import quantize_mamba
from utils.fake_quant_utils.function import QuantizedMatMul
from utils.plot_utils.utils import plot_box_data_perchannel_fig, plot_bar_fig, plot_bar3d_fig

def eval_ppl_(model,test_loader,seqlen=-1,limit=-1):
    nlls = []
    nsamples = test_loader.numel() // seqlen
        
    for i in tqdm(range(nsamples)):
        batch = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
        net_name = model.name.lower() if hasattr(model,"name") else type(model).__name__.lower()
        if "opt" in net_name:
            outputs = model.model.model.decoder(batch)
            hidden_states = outputs[0]
            logits = model.model.lm_head(hidden_states)
        elif "llama" in net_name or "mixtral" in net_name:
            #import pdb;pdb.set_trace()
            outputs = model(batch)
            logits = outputs['logits'];outputs = None
        elif "falcon" in net_name:
            outputs = model.model.transformer(batch)
            hidden_states = outputs[0]
            logits = model.model.lm_head(hidden_states)
        elif "glm" in net_name:
            outputs = model(batch)
            logits = outputs['logits'];outputs = None
        elif "mamba" in net_name:
            outputs = model(batch)
            logits = outputs['logits'];outputs = None
        shift_logits = logits[:, :-1, :]
        shift_labels = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)][
            :, 1:
        ].to(logits.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
        if i == limit:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()

def eval_ppl(args, model, tokenizer,seqlen=2048,limit=-1):
    model.eval()
    
    wiki_testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # wiki_testdata = wiki_testdata.select(range(args.eval_len))
    wiki_testloader = tokenizer("\n\n".join(wiki_testdata["text"]), return_tensors="pt")
    wiki_ppl = eval_ppl_(model, wiki_testloader.input_ids, seqlen, limit)
    print(f'wiki ppl : {wiki_ppl}')

    # lambada_testdata = load_dataset("cimec/lambada", split="validation")
    # lambada_testloader = tokenizer("\n\n".join(lambada_testdata["text"]), return_tensors="pt")
    # lambada_ppl = eval_ppl_(model, lambada_testloader.input_ids, seqlen, limit)
    # print(f'lambada ppl : {lambada_ppl}')

    # c4_testdata = load_dataset(
    #     'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )
    # random.seed(0)
    # valenc = []
    # for _ in range(256):
    #     while True:
    #         i = random.randint(0, len(c4_testdata) - 1)
    #         tmp = tokenizer(c4_testdata[i]['text'], return_tensors='pt')
    #         if tmp.input_ids.shape[1] >= seqlen:
    #             break
    #     i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     valenc.append(tmp.input_ids[:, i:j])
    # c4_testloader = torch.hstack(valenc)
    # c4_ppl = eval_ppl_(model, c4_testloader, seqlen, limit)
    # print(f'c4 ppl : {c4_ppl}')


def analyse_hook(module,input):
    module_name = module_to_name.get(module, "Unnamed module")
    if isinstance(module, (nn.Linear,nn.Conv1d)):
        weight = module.weight.data
        plot_box_data_perchannel_fig(weight,"data/analyse_fig/fp_data/{}_weight_box_data_perchannel.jpg".format(module_name),axis=-1)
        plot_bar_fig(weight, "data/analyse_fig/fp_data/{}_weight_bar_data.jpg".format(module_name))
        plot_bar3d_fig(weight, "data/analyse_fig/fp_data/{}_weight_bar3d_data.jpg".format(module_name))

        temp_input = input[0]
        plot_box_data_perchannel_fig(temp_input,"data/analyse_fig/fp_data/{}_input_box_data_perchannel.jpg".format(module_name),axis=-1)
        plot_bar_fig(temp_input,"data/analyse_fig/fp_data/{}_input_bar_data.jpg".format(module_name))
        plot_bar3d_fig(temp_input,"data/analyse_fig/fp_data/{}_input_bar3d_data.jpg".format(module_name))


def register_hooks(model):
    global module_to_name 
    module_to_name = {module: name for name, module in model.named_modules()}
    for layer in model.backbone.layers:
        layer.mixer.conv1d.register_forward_pre_hook(analyse_hook)
        layer.mixer.in_proj.register_forward_pre_hook(analyse_hook)
        layer.mixer.x_proj.register_forward_pre_hook(analyse_hook)
        layer.mixer.dt_proj.register_forward_pre_hook(analyse_hook)
        layer.mixer.out_proj.register_forward_pre_hook(analyse_hook)

class QuantCfg():
    quantize = False
    # quantize_bmm_input = True
    w_bits = 8
    a_bits = 8
    hs_bits = 8
    matmul_bits = 8
    eval_len = 100

if __name__ == "__main__":
    cfg = QuantCfg
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    model.half().to("cuda")

    if cfg.quantize:
        quantize_mamba(model,a_bits=cfg.a_bits,w_bits=cfg.w_bits)
        torch.matmul = QuantizedMatMul(cfg.matmul_bits).__call__

    register_hooks(model)

    input_ids = tokenizer("Mamba is a type of", return_tensors="pt")["input_ids"]
    out = model.generate(input_ids.to("cuda"), max_new_tokens=10)
    print(tokenizer.batch_decode(out))

    with torch.no_grad():
        eval_ppl(cfg, model, tokenizer)