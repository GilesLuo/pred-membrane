import numpy as np
from tqdm import tqdm
import pandas as pd


def save_pdp_1D(dataframe, save_name, x_name_list=None, y_name_list=None, mode='relative'):
    '''
    :param dataframe: prediction DataFrame
    :param save_name: File save name
    :param x_name_list: None
    :param y_name_list: None
    :param mode: "relative"
    :return: Dataframe Save-files
    '''
    def compute_one_pdp_1D(data, x_name_dp, y_name_dp):
        # The input data must contain x_name_dp, y_name_dp
        x_values = np.unique(data[x_name_dp].to_numpy())
        PD = []
        for x in x_values:
            y_values = data[data[x_name_dp] == x][y_name_dp].to_numpy()
            partial_d = np.concatenate(
                (x.reshape(1, 1), np.mean(y_values).reshape(1, 1), np.max(y_values).reshape(1, 1),
                 np.min(y_values).reshape(1, 1),
                 np.percentile(y_values, [25, 50, 75]).reshape(1, -1)), axis=-1)
            PD.append(partial_d)
        PD = np.concatenate(PD, axis=0)
        y_mean = np.mean(PD[:, 1], axis=0)
        if mode == "relative":
            PD[:, 1:] -= y_mean.repeat(6, axis=-1)
        df = pd.DataFrame(PD)
        df.columns = [x_name_dp, "mean", "max", "min", "quat25", "quat50", "quat75"]
        return df

    if x_name_list is None:
        x_name_list = dataframe.columns[:4]
    if y_name_list is None:
        y_name_list = dataframe.columns[4:]
    writer = pd.ExcelWriter(save_name, engine='openpyxl')
    for y_name in tqdm(y_name_list):
        for x_name in x_name_list:
            df = compute_one_pdp_1D(dataframe, x_name, y_name)
            df.to_excel(writer, sheet_name="{}-{}-1D".format(y_name.strip(), x_name.strip()), startrow=0, index=False)
    writer.save()


def save_pdp_2D(data, save_name, x_name_list=None, y_name_list=None, mode='relative'):
    '''
    :param data: prediction data
    :param save_name: File save name
    :param x_name_list: None
    :param y_name_list: None
    :param mode: "relative"
    :return: Dataframe Save-files
    '''
    def compute_one_pdp_2D(data, x_name_dp, y_name_dp):
        x_values1 = np.unique(data[x_name_dp[0]].to_numpy())
        x_values2 = np.unique(data[x_name_dp[1]].to_numpy())
        PD = []
        for x1 in x_values1:
            for x2 in x_values2:
                y_values = data[(data[x_name_dp[0]] == x1) & (data[x_name_dp[1]] == x2)][y_name_dp].to_numpy()
                partial_d = np.concatenate((x1.reshape(1, 1), x2.reshape(1, 1), np.mean(y_values).reshape(1, 1),
                                            np.max(y_values).reshape(1, 1),
                                            np.min(y_values).reshape(1, 1),
                                            np.percentile(y_values, [25, 50, 75]).reshape(1, -1)), axis=-1)
                PD.append(partial_d)
        PD = np.concatenate(PD, axis=0)
        y_mean = np.mean(PD[:, 2], axis=0)
        if mode == "relative":
            PD[:, 2:] -= y_mean.repeat(6, axis=-1)
        df = pd.DataFrame(PD)
        df.columns = [x_name_dp[0], x_name_dp[1], "mean", "max", "min", "quat25", "quat50", "quat75"]
        return df

    if x_name_list is None:
        x_name_list = []
        for i in data.columns[:4]:
            for j in data.columns[:4]:
                if i != j:
                    x_name_list.append([i, j])
    if y_name_list is None:
        y_name_list = data.columns[4:]
    writer = pd.ExcelWriter(save_name, engine='openpyxl')
    for y_name in tqdm(y_name_list):
        for x_name in x_name_list:
            df = compute_one_pdp_2D(data, x_name, y_name)
            df.to_excel(writer, sheet_name="{}-{}_{}-2D".format(y_name.strip(),
                                                                x_name[0].strip(), x_name[1].strip()), startrow=0,
                        index=False)
    writer.save()

