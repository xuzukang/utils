import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os

# fig, ax = plt.subplots(figsize=(48, 6))  #创建图像并设置画布大小 plt.figure(figsize=figsize)
# plt.tick_params(axis='both', which='major', labelsize=14)  # 设置主刻度标签的大小
# plt.tick_params(axis='both', which='minor', labelsize=10)  # 设置次刻度标签的大小
# ax.set_xticks(index)  # 设置刻度位置
# ax.set_xticklabels(index, rotation=45) #刻度旋转
# plt.tight_layout() #用于自动调整子图参数，使图形中的子图、标签、标题等不重叠，并且整体布局更加紧凑美观
# ax.tick_params(axis='x', labelsize=7)
# ax.vlines(index, ymin=0, ymax=data, colors='gray', linestyles='--')
# plt.title('Box Plot per Channel') #设置标题
# plt.xlabel('Channel') #设置x轴标签
# plt.ylabel('Value') #设置y轴标签


def plot_box_data_perchannel_fig(data_,path,axis=0):
    if isinstance(data_,torch.Tensor):
        if data_.requires_grad:
            data = data_.detach().cpu().numpy()
        else:
            data = data_.cpu().numpy()
    shape = data.shape
    if axis >= len(shape):
        raise ValueError("Axis should be less than data.shape")
    permuted_data = np.moveaxis(data, axis, 0)
    reshaped_data = permuted_data.reshape(shape[axis], -1)
    plt.figure(figsize=(max(shape[axis] // 10, 6),6))
    plt.title(path)
    plt.boxplot(reshaped_data.T)
    plt.xticks(range(0,shape[axis]+1,10))
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout failed with error: {e}")
    plt.savefig(path)
    plt.close()
    print('saving:  ', path)


def plot_bar_fig(data_,path):
    if isinstance(data_,torch.Tensor):
        if data_.requires_grad:
            data = data_.detach().cpu().numpy()
        else:
            data = data_.cpu().numpy()
    data_range = np.max(data) - np.min(data)
    bin_width = data_range / 30  # 设定每个区间的宽度
    bins = np.arange(np.min(data), np.max(data) + bin_width, bin_width)
    plt.hist(data.reshape(-1), bins=bins, edgecolor='black')
    plt.title(path)
    plt.xlabel("x_val")
    plt.ylabel("num")
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout failed with error: {e}") #用于自动调整子图参数，使图形中的子图、标签、标题等不重叠，并且整体布局更加紧凑美观
    plt.savefig(path)
    plt.close()
    print('saving:  ', path)

def plot_bar3d_fig(data_,path):
    if isinstance(data_,torch.Tensor):
        data_ = data_.reshape(data_.shape[0],-1).abs()
        if data_.requires_grad:
            data = data_.detach().cpu().numpy()
        else:
            data = data_.cpu().numpy()
    else:
        data = data_

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 获取数据的维度
    x_len, y_len = data.shape

    # 生成X和Y的坐标
    _x = np.arange(x_len)
    _y = np.arange(y_len)
    
    # meshgrid的参数顺序应该与数据存储的顺序一致：行 -> x轴, 列 -> y轴
    _xx, _yy = np.meshgrid(_x, _y, indexing="ij")
    x, y = _xx.ravel(), _yy.ravel()

    # 数据展开为一维
    top = data.ravel()
    bottom = np.zeros_like(top)
    
    # 调整宽度和深度
    width = depth = 0.2

    # 使用颜色映射工具
    colors = plt.cm.viridis(top / float(top.max()))

    # 绘制3D柱状图
    ax.bar3d(x, y, bottom, width, depth, top, shade=True, color=colors)

    # 设置轴标签和标题
    plt.title(path)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_zlabel("Value")
    
    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(top)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    
    # 调整视角
    ax.view_init(elev=30, azim=45)

    try:
        plt.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout failed with error: {e}")
        
    # 保存图像
    plt.savefig(path)
    plt.close()
    print('saving:  ', path)

def find_images(directory, prefix, suffix):
    """
    在指定目录中查找符合特定前缀和后缀的图片文件。
    
    参数:
    - directory: 图片所在的目录。
    - prefix: 文件名前缀。
    - suffix: 文件名后缀。
    
    返回:
    - 符合条件的图片文件路径列表。
    """
    files = os.listdir(directory)
    selected_files = [os.path.join(directory, f) for f in files if f.startswith(prefix) and f.endswith(suffix)]
    return selected_files
def concat_images(image_paths, images_per_row, save_path):
    """
    将多张图片拼接成一张大图。

    参数:
    - image_paths: 图片路径的列表。
    - images_per_row: 每行图片的数量。
    - save_path: 拼接后的图片保存路径。
    """
    if not image_paths:
        print("没有找到符合条件的图片。")
        return
    
    # 加载第一张图片以获取单张图片的尺寸
    with Image.open(image_paths[0]) as img:
        img_width, img_height = img.size
    
    # 计算大图的尺寸
    rows = (len(image_paths) + images_per_row - 1) // images_per_row
    concat_image = Image.new('RGB', (images_per_row * img_width, rows * img_height))
    
    # 按顺序拼接图片
    for idx, img_path in enumerate(image_paths):
        row = idx // images_per_row
        col = idx % images_per_row
        with Image.open(img_path) as img:
            concat_image.paste(img, (col * img_width, row * img_height))
    
    # 保存拼接后的图片
    concat_image.save(save_path)
    print(f"图片已拼接并保存到 {save_path}")

