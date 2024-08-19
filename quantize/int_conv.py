import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer


class QuantConv1d(nn.Conv1d):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Conv1d,
        weight_quant_params: dict = {"dynamic_method":"per_tensor"},
        act_quant_params: dict = {"dynamic_method":"per_tensor"},
        disable_input_quant=False,
    ):
        super().__init__(org_module.in_channels,
                         org_module.out_channels,
                         kernel_size=org_module.kernel_size,
                         stride=org_module.stride,
                         padding=org_module.padding,
                         dilation=org_module.dilation,
                         bias=org_module.bias is not None)
        self.fwd_kwargs = dict()
        self.fwd_func = F.conv1d
        self.weight=org_module.weight
        if org_module.bias is not None:
            self.bias=org_module.bias
        else:
            self.bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,has_batch_dim=True)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

        self.stride = org_module.stride
        self.padding = org_module.padding
        self.dilation = org_module.dilation
        self.groups = org_module.groups

     
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        
        out = self.fwd_func(
                input, weight, bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                **self.fwd_kwargs)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant



class QuantConv2d(nn.Conv2d):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module,
        weight_quant_params: dict = {"dynamic_method":"per_tensor"},
        act_quant_params: dict = {"dynamic_method":"per_tensor"},
        disable_input_quant=False,
    ):
        super().__init__(org_module.in_channels, org_module.out_channels, org_module.kernel_size,)
        self.fwd_kwargs = dict()
        self.fwd_func = F.conv2d
        self.weight=org_module.weight
        if org_module.bias is not None:
            self.bias=org_module.bias
        else:
            self.bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,has_batch_dim=True)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

        self.in_channels = org_module.in_channels
        self.out_channels = org_module.out_channels
        self.kernel_size = org_module.kernel_size
        self.stride = org_module.stride
        self.padding = org_module.padding
        self.dilation = org_module.dilation
        self.groups = org_module.groups

     
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        
        out = self.fwd_func(
                input, weight, bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                **self.fwd_kwargs)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        

class QuantConv3d(nn.Conv3d):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module,
        weight_quant_params: dict = {"dynamic_method":"per_tensor"},
        act_quant_params: dict = {"dynamic_method":"per_tensor"},
        disable_input_quant=False,
    ):
        super().__init__(org_module.in_channels, org_module.out_channels, org_module.kernel_size,)
        self.fwd_kwargs = dict()
        self.fwd_func = F.conv3d
        self.weight=org_module.weight
        if org_module.bias is not None:
            self.bias=org_module.bias
        else:
            self.bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,has_batch_dim=True)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

        self.in_channels = org_module.in_channels
        self.out_channels = org_module.out_channels
        self.kernel_size = org_module.kernel_size
        self.stride = org_module.stride
        self.padding = org_module.padding
        self.dilation = org_module.dilation
        self.groups = org_module.groups

     
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        
        out = self.fwd_func(
                input, weight, bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                **self.fwd_kwargs)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


