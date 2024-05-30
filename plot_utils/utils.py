import numpy as np
import matplotlib.pyplot as plt


def plot_box_data_perchannel_fig(data,axis,path):
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