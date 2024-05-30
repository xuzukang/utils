
import numpy as np
import torch


def rgb_to_yuv(rgb_image):
    # 转换RGB图像为YUV444格式
    yuv_image = torch.zeros_like(rgb_image)
    
    r = rgb_image[:, 0, :, :]
    g = rgb_image[:, 1, :, :]
    b = rgb_image[:, 2, :, :]

    # 计算Y分量
    y = 0.299 * r + 0.587 * g + 0.114 * b

    # 计算U分量
    u = - 0.168736 * r - 0.331264 * g  + 0.5* b + 128

    # 计算V分量
    v = 0.5 * r - 0.418688 * g -0.081312 * b + 128

    # 将YUV分量存储到YUV图像
    yuv_image[:, 0, :, :] = y
    yuv_image[:, 1, :, :] = u
    yuv_image[:, 2, :, :] = v

    return yuv_image

