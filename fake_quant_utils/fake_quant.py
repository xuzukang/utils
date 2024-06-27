from ast import Dict
import torch
from torch import nn
import numpy as np
from functools import partial
from utils.config.attrdict import AttrDict


quant_config = AttrDict(dict(dim='',
                             observer={'method':'minmax',
                                       "percentile":"0.999999",},
                             n_bit=8,
                       )
                 )
@torch.no_grad()
def Dynamic_quantize(data, quant_cfg=quant_config):
    if not quant_cfg:
        return data
    elif quant_cfg.dim== '': #这视pertensor
        if quant_cfg.observer.method == 'minmax':
            max_val = torch.max(data.abs())
        elif quant_cfg.observer.method == 'percentile':
            max_val = torch.tensor(np.percentile(data.float().abs().cpu().numpy(), 
                                                 quant_cfg.observer.percentile * 100)).to(data)
    else: #pertoken:dim=-2, perchannel:dim=-1
        dim = quant_cfg.dim
        if quant_cfg.observer.method == 'minmax':
            max_val = torch.max(data.abs(),dim=dim)
        elif quant_cfg.observer.method == 'percentile':
            max_val = torch.tensor(np.percentile(data.float().abs().cpu().numpy(), 
                                                 quant_cfg.observer.percentile * 100,
                                                 axis=dim)).to(data)
    q_max = 2 ** (quant_cfg.n_bit - 1) - 1
    scales = max_val.clamp(min=1e-5).div(q_max)
    data = torch.round(data/scales) * (scales)
    return data


class QLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        config=AttrDict(dict(w_cfg=quant_config,
                             i_cfg=quant_config,
                             o_cfg=""))

    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

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

    def to(self, *args, **kwargs):
        super(QLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = Dynamic_quantize(x,self.config.i_cfg)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = Dynamic_quantize(y,self.config.o_cfg)
        return q_y

    @staticmethod
    def from_float(
        module, config=AttrDict(dict(w_cfg=quant_config,
                                     i_cfg=quant_config,
                                     o_cfg=""))
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = QLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            config=config
        )

        new_module.weight.data = Dynamic_quantize(module.weight.data,config.w_cfg)
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"QLinear({self.in_features}, {self.out_features}, bias={self.bias is not None})"


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
        config=AttrDict(dict(w_cfg=quant_config,
                             i_cfg=quant_config,
                             o_cfg=""))
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

        self.config = config

    def to(self, *args, **kwargs):
        super(QConv1d, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = Dynamic_quantize(x, self.config.i_cfg)
        y = torch.functional.F.conv1d(q_x, self.weight, self.bias,
                                      padding=self.padding,
                                      groups=self.groups)
        q_y = Dynamic_quantize(y, self.config.o_cfg)
        return q_y

    @staticmethod
    def from_float(
        module, config=AttrDict(dict(w_cfg=quant_config,
                                     i_cfg=quant_config,
                                     o_cfg=""))
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
            config=config
        )
        new_module.weight.data = Dynamic_quantize(module.weight.data,config.w_cfg)
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"QConv1d({self.in_features}, {self.out_features}, bias={self.bias is not None})"


class QMatMul(nn.Module):
    def __init__(self,
        config=AttrDict(dict(i_cfg=quant_config,
                             o_cfg=""))

    ):
        super().__init__()
        self.config = config


    @torch.no_grad()
    def forward(self, x,y):
        q_x = Dynamic_quantize(x,self.config.i_cfg)
        q_y = Dynamic_quantize(y,self.config.i_cfg)
        out =  q_x @ q_y
        return Dynamic_quantize(out,self.config.o_cfg)

    @staticmethod
    def from_float(
        module, config=AttrDict(dict(i_cfg=quant_config,
                                     o_cfg=""))
    ):
        new_module = QMatMul(
            config=config
        )
        return new_module

    def __repr__(self):
        return f"QMatmul()"


def quantize_vim_torch(
    model, config=AttrDict(dict(w_cfg=quant_config,
                                i_cfg=quant_config,
                                o_cfg="",
                                matmul_cfg=quant_config)),
    quant_blocks=24,
):
    # from model.mamba import MambaBlock
    from vim.vim_torch import MambaBlock

    ori_config = config
    for name, m in model.named_modules():
        if isinstance(m, MambaBlock):
            if int(name.split('.')[1]) >= quant_blocks:
                continue 

            
            # if name.split('.')[1] in ["12",'13','14','15']:
            #     q_cfg = AttrDict(dict(dim='',
            #                  observer={'method':'minmax',
            #                            "percentile":"0.999999",},
            #                  n_bit=6,
            #            )
            #      )
            #     config = AttrDict(dict(w_cfg=q_cfg,
            #                     i_cfg=quant_config,
            #                     o_cfg="",
            #                     matmul_cfg=quant_config))
            # else:
            #     config=ori_config
            
            # if name.split('.')[1] not in ["23",]:
            #     m.in_proj = QLinear.from_float(m.in_proj, config=config)
            m.in_proj = QLinear.from_float(m.in_proj, config=config)
            
            # if name.split('.')[1] not in ["23"]:
            #     m.conv1d = QConv1d.from_float(m.conv1d, config=config)
            m.conv1d = QConv1d.from_float(m.conv1d, config=config)

            # if name.split('.')[1] not in ["23"]:
            #     m.conv1d_b = QConv1d.from_float(m.conv1d_b, config=config)
            m.conv1d_b = QConv1d.from_float(m.conv1d_b, config=config)

            # if name.split('.')[1] not in ["23",]:
            #     m.x_proj = QLinear.from_float(m.x_proj, config=config)
            m.x_proj = QLinear.from_float(m.x_proj, config=config)
            
            # if name.split('.')[1] not in ["23",]:
            #     m.x_proj_b = QLinear.from_float(m.x_proj_b, config=config)
            m.x_proj_b = QLinear.from_float(m.x_proj_b, config=config)
            
            # if name.split('.')[1] not in ["23"]:
            #     m.dt_proj = QLinear.from_float(m.dt_proj, config=config)
            m.dt_proj = QLinear.from_float(m.dt_proj, config=config)

            # if name.split('.')[1] not in ["23"]:
            #     m.dt_proj_b = QLinear.from_float(m.dt_proj_b, config=config)
            m.dt_proj_b = QLinear.from_float(m.dt_proj_b, config=config)
    
            # if name.split('.')[1] not in ["18",'19','23']:
            #     m.out_proj = QLinear.from_float(m.out_proj, config=config)
            m.out_proj = QLinear.from_float(m.out_proj, config=config)

            # if name.split('.')[1] not in ["5",]:
                # m.matmul = QMatMul.from_float(m.matmul, config=config)
            m.matmul = QMatMul.from_float(m.matmul, config=config)

            # if name.split('.')[1] not in ["0",'5','11','23']:
                # m.matmul_b = QMatMul.from_float(m.matmul_b, config=config)
            m.matmul_b = QMatMul.from_float(m.matmul_b, config=config)

    return model
