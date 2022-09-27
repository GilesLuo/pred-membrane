import copy

import numpy as np
import pandas as pd

column_names = ['CPIP', 'CTMC', 'CSDS', 'ti',
                'Na2SO4', 'MgCl2', 'NaCl', 'PEG400', 'PWP',
                'Na2SO4/NaCl', 'MgCl2/NaCl', 'PEG/NaCl']

# Preprocessing of data
def preprocessing(data):
    data_ = copy.deepcopy(data)
    # preprocessing
    data_[:, 1] = np.log(data_[:, 1])
    data_[:, 2] = data_[:, 2] / 2.05
    data_[data_ == 180.] = 120.  # we consider t=180 the same as t=120 since the rejection difference is minor
    data_[:, 3] /= 60.
    data_[:, 4:-3] *= 0.01
    data_[:, -3:] = np.log(data_[:, -3:])
    return data_


def random_pair_split(data, proportions):
    '''
    @param data: input data
    @param proportions: Allocation proportion
    @return: The data set is partitioned to a specified proportion
    '''
    if sum(proportions) != 1:
        raise ValueError("dataset split should sum up to 1")
    splitted_datasets = []
    unique_samples = np.unique(data[:, :4], axis=0)
    indices = np.random.permutation(unique_samples.shape[0])
    sizes = [int(unique_samples.shape[0] * p) for p in proportions]
    sizes[-1] += unique_samples.shape[0] - sum(sizes)
    for size in sizes:
        seg_idx = indices[:size]
        indices = indices[size:]
        set = data[np.concatenate([2 * seg_idx, 2 * seg_idx + 1])]
        splitted_datasets.append(set)
    assert sum(sizes) * 2 == data.shape[0]
    return splitted_datasets

def load_xlsx(file_path) -> np.array:
    data = pd.read_excel(file_path, header=None, engine="openpyxl")
    data = data.iloc[:, :12].dropna()
    column_name = list(data)
    data = data.to_numpy()
    return data, column_name


def get_param_size(net):
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return trainable_num


def augment_mean(data):
    unq_in_features, labels = np.unique(data[:, :4], axis=0), data[:, 4:]

    # get mean of labels
    sample_set = []
    for i in range(2):
        sample_set.append(labels[i::2])
    sample_set = np.stack(sample_set)  # [d_t, num_data, num_labels]
    label_mean = np.mean(sample_set, axis=0)
    return np.concatenate([unq_in_features, label_mean], axis=1)


