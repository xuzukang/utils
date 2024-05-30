import torch

class QuantizedMatMul:
    def __init__(self, num_bits=8, quant_mode="per_tensor"):
        self.num_bits = num_bits
        self.quant_mode = quant_mode
        
    def quantize_per_channel_absmax(self, w, num_bits=8):
        # w: (out_features, in_features)
        scales = w.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (num_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        w.div_(scales).round_().mul_(scales)
        return w

    def quantize_per_token_absmax(self,t, num_bits=8):
        t_shape = t.shape
        t.reshape(-1, t_shape[-1])
        scales = t.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (num_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        t.div_(scales).round_().mul_(scales)
        return t

    # @torch.no_grad()
    def quantize_per_tensor_absmax(self, w, num_bits=8):
        # w: (out_features, in_features)
        scales = w.abs().max()
        q_max = 2 ** (num_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        w.div_(scales).round_().mul_(scales)
        return w

    def __call__(self, A, B):
        if self.quant_mode == "per_channel":
            A_q = self.quantize_per_channel_absmax(A,self.num_bits)
            B_q = self.quantize_per_channel_absmax(B,self.num_bits)
        elif self.quant_mode == "per_tensor":
            A_q = self.quantize_per_tensor_absmax(A,self.num_bits)
            B_q = self.quantize_per_tensor_absmax(B,self.num_bits)
        elif self.quant_mode == "per_token":
            A_q = self.quantize_per_token_absmax(A,self.num_bits)
            B_q = self.quantize_per_token_absmax(B,self.num_bits)
        else:
            raise ValueError("Unsupported quantization mode: {}".format(self.quant_mode))

        result_q = A_q@B_q
        return result_q


if __name__ == "__main__":
    A = torch.randn(2, 3)
    B = torch.randn(3, 2)
    num_bits = 8

    qmm = QuantizedMatMul(num_bits)
    result = qmm(A, B)
    print("量化矩阵乘法结果：", result)