import numpy as np
import torch


def get_feature_importance(dataframe, model, loss_fn, repeat_time, save_dir, mode, show_fig=True):
    '''
    :param dataframe: Store the result
    :param model: model
    :param loss_fn: loss function
    :param repeat_time: repeat time
    :param save_dir: save_file path
    :param mode: 'relative'
    :param show_fig: visualization
    :return: The result of the importance of each feature
    '''
    assert mode in ["relative", "absolute"]
    model.give_importance = False
    column_names = list(dataframe)
    data = dataframe.to_numpy()
    X, y = data[:, :4], data[:, 4:]
    X[:, 1] = np.log(X[:, 1] * 100)
    X[:, 2] = X[:, 2] / 2.05 * 0.1
    # self.data[self.data == 180.] = 120
    X[:, 3] /= (60. * 4)
    y[:, :5] *= 0.01
    size = y.shape[0]
    importance = []
    with torch.no_grad():
        for input_id in range(4):
            test_loss = 0
            for _ in range(repeat_time):
                X_ = X.copy()

                np.random.shuffle(X_[:, input_id])
                pred = model(torch.tensor(X_, dtype=torch.float))
                loss = loss_fn(pred[:, :5].squeeze(), torch.tensor(y[:, :5], dtype=torch.float).squeeze())
                test_loss += loss.item()
            importance.append({'feature': column_names[input_id], 'loss': test_loss / size * repeat_time})
    df = pd.DataFrame(importance)
    df = df.sort_values('loss')

    if mode == "relative":
        import torch.nn as nn
        importance = df.loss.to_numpy()
        importance /= np.linalg.norm(importance)
        weighted_importance = nn.Softmax(dim=-1)(torch.from_numpy(importance))
        df.loss = weighted_importance.numpy()
        plt.ylabel = 'weight'
        plt.figure()
        plt.barh(np.arange(4), df.loss)
        plt.yticks(np.arange(4), df.feature.values)
        plt.title("Feature importance")
    else:
        plt.ylabel = 'loss'
        plt.figure()
        plt.barh(np.arange(4), df.loss)
        plt.yticks(np.arange(4), df.feature.values)
        plt.title("Feature importance " + mode)

    plt.savefig(save_dir + 'prediction/feature_importance-{}.jpg'.format(mode))
    if show_fig:
        plt.show()
    return df


if __name__ == "__main__":
    import torch.nn as nn
    import pandas as pd
    from matplotlib import pyplot as plt

    # Enter the parameters in order to get the corresponding feature importance dataframe
    file_path = "./data/Experiment data20211023-With calculated selectivity.xlsx"
    data = pd.read_excel(file_path, engine="openpyxl")
    save_dir = 'input-attention/'
    model = torch.load(save_dir + "model/model-{}-3000.pth".format(save_dir[:-1]))
    model.give_importance = False
    importance = get_feature_importance(data, model, nn.MSELoss(), mode='relative', repeat_time=10, save_dir=save_dir)
