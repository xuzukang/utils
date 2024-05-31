import torch
from torch import nn
from functools import partial


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, method="minmax",percentile=0.99999,n_bits=8):
    # w: (out_features, in_features)
    if method == 'minmax':
        scales = w.abs().max(dim=-1, keepdim=True)[0]
    elif method == 'percentile':
        scales = torch.quantile(w.float().abs(), percentile, dim=-1, keepdim=True)
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, method="minmax",percentile=0.99999, n_bits=8):
    # w: (out_features, in_features)
    if method == 'minmax':
        scales = w.abs().max()
    elif method == 'percentile':
        scales = torch.quantile(w.float().abs(), percentile)
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, method="minmax",percentile=0.99999, n_bits=8):
    t_shape = t.shape
    t.reshape(-1, t_shape[-1])
    if method == 'minmax':
        scales = t.abs().max(dim=-1, keepdim=True)[0]
    elif method == 'percentile':
        scales = torch.quantile(t.float().abs(), percentile, dim=-1, keepdim=True)
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, method="minmax",percentile=0.99999, n_bits=8):
    t_shape = t.shape
    t.reshape(-1, t_shape[-1])
    if method == 'minmax':
        scales = t.abs().max()
    elif method == 'percentile':
        scales = torch.quantile(t.float().abs(), percentile)
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class QLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        a_method="minmax",
        w_method="minmax",
        w_bits=8,
        a_bits=8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=a_bits,method=a_method)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=a_bits,method=a_method)
        elif act_quant == "per_channel":
            self.act_quant_name = "per_channel"
            self.act_quant = partial(quantize_weight_per_channel_absmax, n_bits=a_bits,method=a_method)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(QLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False,
        a_bits=8,w_bits=8,a_method="minmax",w_method="minmax",
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = QLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            a_method=a_method,
            w_method=w_method,
            a_bits=a_bits,w_bits=w_bits,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=w_bits,method=w_method)  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=w_bits,method=w_method)
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"QLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


class QConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=0,
        groups: int = 1,
        bias: bool = True,
        act_quant: str = "per_token",
        quantize_output: bool = False,
        a_method="minmax",
        w_method="minmax",
        a_bits: int = 8,
        w_bits: int = 8
    ):
        super(QConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups, 
            bias=bias)

        self.weight = nn.Parameter(
            torch.randn(
                out_channels, 
                in_channels // self.groups,  # 注意分组卷积参数
                kernel_size
            ).to(dtype=torch.float16),  # 设置数据类型
            requires_grad=False  # 设置为True，以便于权重更新
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels).to(dtype=torch.float16),
                requires_grad=False
            )
        else:
            self.bias = None

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=a_bits,method=a_method)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=a_bits,method=a_method)
        elif act_quant == "per_channel":
            self.act_quant_name = "per_channel"
            self.act_quant = partial(quantize_weight_per_channel_absmax, n_bits=a_bits,method=a_method)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(QConv1d, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.conv1d(q_x, self.weight, self.bias,
                                      padding=self.padding,
                                      groups=self.groups)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False,
        a_bits=8,w_bits=8,a_method="minmax",w_method="minmax",
    ):
        assert isinstance(module, torch.nn.Conv1d)
        new_module = QConv1d(
            in_channels=module.in_channels, 
            out_channels=module.out_channels, 
            kernel_size=module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size, 
            stride=module.stride,
            padding=module.padding,
            groups=module.groups,
            bias=module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            a_method=a_method,
            w_method=w_method,
            a_bits=a_bits,w_bits=w_bits,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=w_bits,method=w_method)  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=w_bits,method=w_method)
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"QConv1d({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_mamba(
    model, weight_quant="per_tensor", act_quant="per_token", quantize_bmm_input=True,
    a_bits=8,w_bits=8,a_method="minmax", w_method="minmax",
):
    # from model.mamba import MambaBlock
    from transformers.models.mamba.modeling_mamba import MambaMixer

    for name, m in model.named_modules():
        if isinstance(m, MambaMixer):
            m.in_proj = QLinear.from_float(
                m.in_proj, weight_quant=weight_quant, act_quant=act_quant,
                a_bits=a_bits,w_bits=w_bits,
                a_method=a_method,
                w_method=w_method,
            )
            m.conv1d = QConv1d.from_float(
                m.conv1d, weight_quant=weight_quant, 
                act_quant=act_quant,a_bits=a_bits,w_bits=w_bits,
                a_method=a_method,
                w_method=w_method,
            )
            m.x_proj = QLinear.from_float( #影响大
                m.x_proj, weight_quant=weight_quant, 
                act_quant=act_quant,a_bits=a_bits,w_bits=w_bits,
                a_method=a_method,
                w_method=w_method,
            )
            m.dt_proj = QLinear.from_float(
                m.dt_proj, weight_quant=weight_quant, 
                act_quant=act_quant,a_bits=a_bits,w_bits=w_bits,
                a_method=a_method,
                w_method=w_method,
            )
            m.out_proj = QLinear.from_float(  #影响大
                m.out_proj, weight_quant=weight_quant, 
                act_quant=act_quant,a_bits=a_bits,w_bits=w_bits,
                a_method=a_method,
                w_method=w_method,
            )
    return model


def quantize_opt(
    model, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=True
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = QLinear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant
            )
            m.fc2 = QLinear.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = QLinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = QLinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = QLinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.out_proj = QLinear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_llama_like(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = QLinear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.up_proj = QLinear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.down_proj = QLinear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = QLinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = QLinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = QLinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = QLinear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_mixtral(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            m.w1 = QLinear.from_float(
                m.w1, weight_quant=weight_quant, act_quant=act_quant
            )
            m.w2 = QLinear.from_float(
                m.w2, weight_quant=weight_quant, act_quant=act_quant
            )
            m.w3 = QLinear.from_float(
                m.w3, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = QLinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = QLinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = QLinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = QLinear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = QLinear.from_float(
                m.gate, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_falcon(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = QLinear.from_float(
                m.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant
            )
            m.dense_4h_to_h = QLinear.from_float(
                m.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = QLinear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.dense = QLinear.from_float(
                m.dense, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    if isinstance(model, OPTPreTrainedModel):
        return quantize_opt(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")