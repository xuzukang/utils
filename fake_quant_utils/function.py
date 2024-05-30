import torch

class QuantizedMatMul:
    def __init__(self, num_bits):
        self.num_bits = num_bits
        
    def quantize_weight_per_channel_absmax(self, w):
        # w: (out_features, in_features)
        scales = w.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (self.num_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        w.div_(scales).round_().mul_(scales)
        return w


    @torch.no_grad()
    def quantize_weight_per_tensor_absmax(self, w):
        # w: (out_features, in_features)
        scales = w.abs().max()
        q_max = 2 ** (self.num_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        w.div_(scales).round_().mul_(scales)
        return w

    def __call__(self, A, B):
        A_q = self.quantize_weight_per_tensor_absmax(A)
        B_q = self.quantize_weight_per_tensor_absmax(B)
        result_q = A_q@B_q
        return result_q

def quantize_weight_per_channel_absmax(w,num_bits):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (num_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w,num_bits):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (num_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

def Qmatmul( A, B):
    A_q = quantize_weight_per_tensor_absmax(A,8)
    B_q = quantize_weight_per_tensor_absmax(B,8)
    result_q = torch.matmul(A_q, B_q)
    return result_q

if __name__ == "__main__":
    A = torch.randn(2, 3)
    B = torch.randn(3, 2)
    num_bits = 8

    qmm = QuantizedMatMul(num_bits)
    result = qmm(A, B)
    print("量化矩阵乘法结果：", result)