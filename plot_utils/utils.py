import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
from mpl_toolkits.mplot3d import Axes3D

# fig, ax = plt.subplots(figsize=(48, 6))  #创建图像并设置画布大小
# ax.set_xticks(index)  # 设置刻度位置
# ax.set_xticklabels(index, rotation=45) #刻度旋转
# plt.tight_layout() #用于自动调整子图参数，使图形中的子图、标签、标题等不重叠，并且整体布局更加紧凑美观
# ax.tick_params(axis='x', labelsize=7)
# ax.vlines(index, ymin=0, ymax=data, colors='gray', linestyles='--')

def plot_box_data_perchannel_fig(data=np.random.randn(10,10),axis=0,path="./tmp.jpg"):
    shape = data.shape
    if axis >= len(shape):
        raise ValueError("Axis should be less than data.shape")
    permuted_data = np.moveaxis(data, axis, 0)
    reshaped_data = permuted_data.reshape(shape[axis], -1).detach().numpy()
    max_value = np.amax(reshaped_data, axis=-1)
    min_value = np.amin(reshaped_data, axis=-1)
    plt.boxplot(reshaped_data)
    plt.savefig(path)
    plt.close()
    print('max_value', max_value)
    print('min_value', min_value)


def plot_bar_fig(data=np.random.randn(10,10),axis=0,path="./tmp.jpg"):
    data_range = np.max(data) - np.min(data)
    bin_width = data_range / 30  # 设定每个区间的宽度
    bins = np.arange(np.min(data), np.max(data) + bin_width, bin_width)
    fig, ax = plt.subplots(figsize=(48, 6))
    ax.hist(data, bins=bins, edgecolor='black')
    ax.set_title('hist plot')
    ax.set_xlabel("x_val")
    ax.set_ylabel("num")
    plt.tight_layout() #用于自动调整子图参数，使图形中的子图、标签、标题等不重叠，并且整体布局更加紧凑美观
    plt.savefig(path)


def plot_bar3d_fig(data=np.random.randn(10,10),axis=0,path="./tmp.jpg"):
    """
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 获取数据的维度
    x_len, y_len = data.shape

    # 生成X和Y的坐标
    _x = np.arange(x_len)
    _y = np.arange(y_len)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    # 数据展开为一维
    top = data.ravel()
    bottom = np.zeros_like(top)
    width = depth = 1

    # 绘制3D柱状图
    ax.bar3d(x, y, bottom, width, depth, top, shade=True)

    # 设置轴标签和标题
    ax.set_title("bar3d plot")
    ax.set_xlabel("dim0")
    ax.set_ylabel("dim1")
    ax.set_zlabel("value")

    plt.savefig(path)


