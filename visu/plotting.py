import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from utils import column_names


def visu_correlation(model, train_set, test_set, save_dir, device):
    '''
    @param model: model
    @param train_set: training dataset
    @param test_set: testing dataset
    @param save_dir: File saving path
    @param device: device
    @return: Visual correlation coefficient
    '''
    tr_visuloader = DataLoader(train_set, 1000, shuffle=True)
    te_visuloader = DataLoader(test_set, 1000, shuffle=True)

    Pred_train, Y_train = [], []
    Pred_test, Y_test = [], []

    with torch.no_grad():
        for idx, (X_train, y_train) in enumerate(tr_visuloader):
            X_train = X_train.to(device)
            y_pred = model(X_train)
            Pred_train.append(y_pred.to('cpu'))
            Y_train.append(y_train.to("cpu"))

        for idx, (X_test, y_test) in enumerate(te_visuloader):
            X_test = X_test.to(device)
            y_pred = model(X_test)
            Pred_test.append(y_pred.to('cpu'))
            Y_test.append(y_test.to("cpu"))

    Pred_train = torch.cat(Pred_train).numpy()
    Y_train = torch.cat(Y_train).numpy()
    Pred_test = torch.cat(Pred_test).numpy()
    Y_test = torch.cat(Y_test).numpy()

    for i in range(8):
        plt.figure()
        if i < 5:
            max_Y = 1
        else:
            max_Y = max([Y_train[:, i].max(), Pred_train[:, i].max(),
                         Y_test[:, i].max(), Pred_test[:, i].max()])
        plt.plot([0, max_Y], [0, max_Y])
        plt.scatter(Y_train[:, i], Pred_train[:, i], c='blue', alpha=0.6)
        plt.scatter(Y_test[:, i], Pred_test[:, i], c='red', alpha=0.6)
        label = ['reference line', 'training point', 'testing point']
        plt.legend(label, loc='lower right')
        plt.title("correlation-" + str(column_names[4 + i]))
        plt.xlabel("real")
        plt.ylabel("pred")
        try:
            plt.text(0.1 * max_Y, 1 * max_Y,
                     'R^2 train:{:.2}'.format(r2_score(Y_train[:, i], Pred_train[:, i])))
            plt.text(0.1 * max_Y, 0.95 * max_Y,
                     'R^2 test:{:.2}'.format(r2_score(Y_test[:, i], Pred_test[:, i])))
        except ValueError:
            pass
        plt.savefig(os.path.join(save_dir, "correlation-" + str(column_names[4 + i]).replace('/', '／') + '.jpg'))
        plt.close()
    # The last dimension is concatenated by np.concatenate(), and columns of DataFrame are marked
    df_train = pd.DataFrame(np.concatenate([Pred_train, Y_train], axis=-1),
                            columns=["output1", "output2", "output3", "output4", "output5", "output6", "output7",
                                     "output8", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"])
    df_test = pd.DataFrame(np.concatenate([Pred_test, Y_test], axis=-1),
                           columns=["output1", "output2", "output3", "output4", "output5", "output6", "output7",
                                    "output8", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"])
    # Writes the DataFrame to a file
    writer = pd.ExcelWriter(os.path.join(save_dir, "pred_train-test.xlsx"), engine='openpyxl')
    df_train.to_excel(writer, sheet_name="train",
                      startrow=0, index=False)
    df_test.to_excel(writer, sheet_name="test",
                     startrow=0, index=False)
    writer.save()
    return df_train, df_test


def plot_input_freq(input_array, target_input_id, title=None):
    '''
    :param input_array: input array
    :param target_input_id: Given the input array target id
    :param title: None
    :return: plotting
    '''
    entity_list = np.unique(input_array[:, target_input_id])
    freq_count = []
    for i in range(len(entity_list)):
        target_col = np.where(input_array[:, target_input_id] == entity_list[i])
        freq_count.append(len(input_array[target_col]))
    plt.bar(entity_list, freq_count, width=0.05 * max(entity_list))
    if title is not None:
        plt.title(title)
    plt.show()


def plot_input_freq_compare(input_array_old, input_array_new, target_input_id, title=None):
    '''
    :param input_array_old: old input array
    :param input_array_new: new input array
    :param target_input_id: Given the input array target id
    :param title: None
    :return: plotting
    '''
    entity_list = np.unique(input_array_old[:, target_input_id])
    freq_count1 = []
    freq_count2 = []
    for i in range(len(entity_list)):
        target_col1 = np.where(input_array_old[:, target_input_id] == entity_list[i])
        target_col2 = np.where(input_array_new[:, target_input_id] == entity_list[i])
        freq_count1.append(len(input_array_old[target_col1]))
        freq_count2.append(len(input_array_new[target_col2]) - len(input_array_old[target_col1]))
    plt.bar(entity_list, freq_count1, width=0.05 * max(entity_list))
    plt.bar(entity_list, freq_count2, bottom=freq_count1, width=0.05 * max(entity_list))
    if title is not None:
        plt.title(title)
    plt.show()


def plot_label_freq_compare(label_data_old, label_data_new, y_range, num_classes=240, title=None, xlabel=None,
                            ylabel=None):
    '''
    :param label_data_old: Old labeled data
    :param label_data_new: New labeled data
    :param y_range: The range of y
    :param num_classes: classes number(Step length)
    :param title: None
    :param xlabel: None
    :param ylabel: None
    :return: plotting
    '''
    y_min, y_max = y_range
    classes = np.linspace(y_min, y_max, num=num_classes + 1)

    freq_list_old = []
    freq_list_new = []
    for i in range(1, num_classes + 1):
        freq1 = np.logical_and(label_data_old < classes[i], label_data_old > classes[i - 1])
        freq_list_old.append(freq1.sum())

        freq2 = np.logical_and(label_data_new < classes[i], label_data_new > classes[i - 1])
        freq_list_new.append(freq2.sum() - freq1.sum())

    width_interval = (y_max - y_min) / (num_classes * 1.5)
    plt.bar(classes[1:], freq_list_old, color='black', width=width_interval)
    plt.bar(classes[1:], freq_list_new, bottom=freq_list_old, color='red', width=width_interval)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


def plot_label_freq(label_data, y_range, num_classes=240, title=None, xlabel=None, ylabel=None):
    '''
    :param label_data: data label
    :param y_range: The range of y
    :param num_classes: classes number(Step length)
    :param title: None
    :param xlabel: None
    :param ylabel: None
    :return: plotting
    '''
    y_min, y_max = y_range
    classes = np.linspace(y_min, y_max, num=num_classes + 1)

    freq_list = []
    for i in range(1, num_classes + 1):
        freq = np.logical_and(label_data < classes[i], label_data > classes[i - 1])
        freq_list.append(freq.sum())

    plt.bar(classes[1:], freq_list, width=(y_max - y_min) / (num_classes * 1.1))
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


def plot_partial2D(data: np.array):
    # Determine whether the shape of the input data meets the requirements
    assert len(data.shape) == 2
    for i in range(data.shape[0]):
        plt.plot(data[i, :-1], data[i, -1])
    plt.show()


def plot_partial3D(data: np.array):
    # Determine whether the shape of the input data meets the requirements
    assert len(data.shape) == 2 and data.shape[1] == 3
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(data[:, 0], data[:, 1], data[:, 2])
    plt.show()


if __name__ == "__main__":
    pass
