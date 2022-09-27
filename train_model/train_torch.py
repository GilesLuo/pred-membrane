import copy
import openpyxl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import math
import os
import torch.nn as nn
from utils import load_xlsx, random_pair_split, preprocessing
from train_model.torch_models import MLP, MLP_skip
from sklearn import metrics
from torch.utils.data import Dataset
import torch
from visu.plotting import visu_correlation
from matplotlib import pyplot as plt

l = nn.HuberLoss()


def get_correlation(model, train_set, test_set, device):
    """
    @param model:Model
    @param train_set:training dataset
    @param test_set: testing dataset
    @param device: device
    @return:A data set of correlation coefficients
    """
    tr_visuloader = DataLoader(train_set, 1000)
    te_visuloader = DataLoader(test_set, 1000)
    # The data is loaded by Dataloader, BatchSize is set to 1000, and each epoch is read in sequence

    Pred_train, Y_train = [], []
    Pred_test, Y_test = [], []
    # Create an empty list of train and test data for storage

    # Requires_grad automatically defaults to false without the backpropagation algorithm
    with torch.no_grad():
        for X_train, y_train in tr_visuloader:
            y_pred = model(X_train.to(device))
            Pred_train.append(y_pred.to("cpu"))
            Y_train.append(y_train)
        # Use model to train the model data Y_train
        for X_test, y_test in te_visuloader:
            y_pred = model(X_test.to(device))
            Pred_test.append(y_pred.to("cpu"))
            Y_test.append(y_test)
        # Model data trained using test data y_pred

    # Concatenate rows of all the list data and convert it to an array operation
    Pred_train = torch.cat(Pred_train, dim=0).numpy()
    Y_train = torch.cat(Y_train, dim=0).numpy()
    Pred_test = torch.cat(Pred_test, dim=0).numpy()
    Y_test = torch.cat(Y_test, dim=0).numpy()

    # The correlation index R2 of Y_train and Pred_train, Y_test and Pred_test were calculated respectively,
    # and the fitting degree of the regression was calculated
    r2 = []
    for i in range(8):
        if np.any(np.isnan(Pred_train)) or np.any(np.isnan(Pred_test)):
            r2.append(0.)
            r2.append(0.)
        else:
            r2.append(metrics.r2_score(Y_train[:, i], Pred_train[:, i]))
            r2.append(metrics.r2_score(Y_test[:, i], Pred_test[:, i]))
    # Pd.dataframe () was used to encapsulate the correlation coefficients of various membranes,
    # and index identification was carried out
    df = pd.DataFrame(np.array(r2).reshape(-1, 16))
    df.columns = ['R2_train_Na2SO4', 'R2_test_Na2SO4',
                  'R2_train_MgCl2', 'R2_test_MgCl2',
                  'R2_train_NaCl', 'R2_test_NaCl',
                  'R2_train_PEG400', 'R2_test_PEG400',
                  'R2_train_PWP', 'R2_test_PWP',
                  'R2_train_sel1', 'R2_test_sel1',
                  'R2_train_sel2', 'R2_test_sel2',
                  'R2_train_sel3', 'R2_test_sel3',
                  ]
    df["test_sel_MSE"] = ((Pred_test[:, 5:8] - Y_test[:, 5:8]) ** 2).mean()
    df["train_sel_MSE"] = ((Pred_train[:, 5:8] - Y_train[:, 5:8]) ** 2).mean()
    return df


class MembraneSet(Dataset):
    def __init__(self, data: np.array, input_size=4):
        '''
        @param data: all data [X, Y]
        @param input_size: 3 or 4
        @param gen: bool
        '''
        self.output_id = [4, 5, 6, 7, 8, 9, 10, 11]
        self.input_size = input_size
        self.data = preprocessing(data)

    # The data in the dataset is called by index
    def __getitem__(self, index):
        item = self.data[index]
        X = item[:self.input_size]
        y = item[self.output_id]
        # print(output)
        return torch.tensor(X).to(torch.float32), torch.tensor(y).to(torch.float32)

    def __len__(self):
        return self.data.shape[0]


# A function that calculates the difference between the label value and the predicted value
def loss_fn(use_reg):
    assert use_reg >= 0

    def loss_without_sel(pred: torch.tensor, y: torch.tensor):
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(dim=0)
        if len(y.shape) == 1:
            y = y.unsqueeze(dim=0)
        y_rej, y_sel = y[:, :5], y[:, 5:]
        pred_rej, pred_sel = pred[:, :5], pred[:, 5:]
        MSE_rej = l(pred_rej, y_rej)
        return MSE_rej, 0

    def loss_with_sel(pred: torch.tensor, y: torch.tensor):
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(dim=0)
        if len(y.shape) == 1:
            y = y.unsqueeze(dim=0)
        y_rej, y_sel = y[:, :5], y[:, 5:]
        pred_rej, pred_sel = pred[:, :5], pred[:, 5:]
        MSE_rej = torch.mean((pred_rej - y_rej) ** 2)
        MSE_sel = torch.mean((pred_sel - y_sel) ** 2)
        MSE_sel = MSE_sel / MSE_sel.detach()

        return (1 - use_reg) * MSE_rej, use_reg * MSE_sel * MSE_rej.detach()

    if use_reg > 0:
        return loss_with_sel
    else:
        return loss_without_sel


def train_loop(dataloader, model, loss_fn, optimizer, device):
    '''
    @param dataloader: data loading
    @param model: model
    @param loss_fn: Loss function calculation
    @param optimizer: Optimization function
    @param device: device
    @return: Loss value
    '''
    size = len(dataloader)
    loss_to_show = [0, 0]
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss1, loss2 = loss_fn(pred, y)
        loss = loss1 + loss2

        optimizer.zero_grad()  # Clears the computational effects of the previous gradient
        loss.backward()
        optimizer.step()
        loss_to_show[0] += loss1.item()
        try:
            loss_to_show[1] += float(torch.mean((pred[5:] - y[5:]) ** 2))
        except AttributeError:
            pass
    loss_to_show[0] /= size
    loss_to_show[1] /= size
    return loss_to_show


def test_loop(dataloader, model, device):
    '''
    @param dataloader: data loading
    @param model: model
    @param device: device
    @return: Return the trained neural network to calculate the loss value of the test data
    '''
    preds, Y = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            if torch.any(torch.isnan(pred)):
                return 1e6, 1e6
            preds.append(pred)
            Y.append(y)
    preds = torch.cat(preds)
    Y = torch.cat(Y)

    rej_loss = ((preds[:, :5] - Y[:, :5]) ** 2).mean(1).mean()
    sel_loss = ((preds[:, 5:8] - Y[:, 5:8]) ** 2).mean(1).mean()
    return rej_loss.item(), sel_loss.item()


def training_routine(train_set, test_set, model_type,
                     save_dir, device,
                     use_reg, batch_size, lr, seed, neurons=128,
                     num_epoch=2000, show_plot=False, show_loss=True, save_models=True,
                     ):
    '''
    @param train_set: training dataset
    @param test_set: testing dataset
    @param model_type: model type
    @param save_dir: File saving path
    @param device: device
    @param use_reg: use_reg is a hyperparameter to set the proportion of regularization regarding the norm of main loss.
    @param batch_size: The number of data read in each batch
    @param lr: learning rate
    @param seed: random seed
    @param neurons: Set the number of neurons
    @param num_epoch: The number of epoch
    @param show_plot: Display graphics
    @param show_loss: Show loss value
    @param save_models: Save the trained model
    @return: Return a DataFrame containing train_data, R2 of test_data, loss, and the epoch training times of the iteration
    '''
    # Generate the specified random number seed, convenient to reproduce the experimental results
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Different kinds of MLP-models are selected for training
    if model_type == "mlp":
        model = MLP(4, 5, neurons=neurons).to(device)
    elif model_type == "mlp_skip":
        model = MLP_skip(4, 5).to(device)
    else:
        raise ValueError
    loss = loss_fn(use_reg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)

    train_set = MembraneSet(copy.deepcopy(train_set))
    test_set = MembraneSet(copy.deepcopy(test_set))
    tr_dataloader = DataLoader(train_set, batch_size, shuffle=True)
    te_dataloader = DataLoader(test_set, 1000, shuffle=True)
    # Use lists to record the training results
    epoch = []
    tr_loss, te_loss = [], []
    tr_sel_loss, te_sel_loss = [], []
    tr_r2, te_r2 = [], []
    tr_sel_r2, te_sel_r2 = [], []
    log = []
    for t in range(1, num_epoch + 1):
        tr_l, tr_r = train_loop(tr_dataloader, model, loss, optimizer, device)
        te_l, te_r = test_loop(te_dataloader, model, device)
        df = get_correlation(model, train_set, test_set, device)
        train_R2 = df[['R2_train_Na2SO4', 'R2_train_MgCl2', 'R2_train_NaCl', 'R2_train_PEG400', 'R2_train_PWP']]
        train_sel_R2 = df[['R2_train_sel1', 'R2_train_sel2', 'R2_train_sel3', ]]

        test_R2 = df[['R2_test_Na2SO4', 'R2_test_MgCl2', 'R2_test_NaCl', 'R2_test_PEG400', 'R2_test_PWP', ]]
        test_sel_R2 = df[['R2_test_sel1', 'R2_test_sel2', 'R2_test_sel3', ]]
        scheduler.step()
        if t % 20 == 0 and show_loss:
            print("epoch{}: training loss{:.3}-{:.3}, testing loss{:.3}-{:.3}".format(t, tr_l, tr_r, te_l, te_r))
        if t % 1 == 0 and save_models:
            torch.save(model, os.path.join(save_dir, f"model/epoch{t}.pth"))
            log.append("epoch{}: training loss{:.3}-{:.3}, testing loss{:.3}-{:.3}".format(t, tr_l, tr_r, te_l, te_r))
        epoch.append(t)
        tr_loss.append(tr_l)
        tr_r2.append(float(train_R2.mean(1)))
        te_loss.append(te_l)
        te_r2.append(float(test_R2.mean(1)))

        tr_sel_loss.append(float(df["train_sel_MSE"]))
        tr_sel_r2.append(float(train_sel_R2.mean(1)))
        te_sel_loss.append(float(df["test_sel_MSE"]))
        te_sel_r2.append(float(test_sel_R2.mean(1)))
        if math.isnan(tr_l) or math.isnan(te_l):
            break

    df = pd.DataFrame.from_dict({"train_MSE": tr_loss,
                                 "train_R2": tr_r2,
                                 "test_MSE": te_loss,
                                 "test_R2": te_r2,
                                 "train_sel_MSE": tr_sel_loss,
                                 "train_sel_R2": tr_sel_r2,
                                 "test_sel_MSE": te_sel_loss,
                                 "test_sel_R2": te_sel_r2,
                                 "epoch": epoch
                                 })
    df.to_excel(os.path.join(save_dir, 'training_log.xlsx'))
    best_epoch = df.loc[df["test_sel_MSE"].idxmin(), "epoch"]
    if save_models:
        model = torch.load(os.path.join(save_dir, "model", f"epoch{best_epoch}.pth"))
        visu_correlation(model, train_set, test_set, os.path.join(save_dir, "img"), device)
        if show_plot:
            plt.show()
        else:
            plt.close("all")
    return df


if __name__ == "__main__":
    from math import ceil
    from tqdm import tqdm
    import time
    import random

    # The corresponding hyperparameters are given initial values
    # And the neural network is trained through for loop to get the results, and the results are visualized
    seeds = range(1, 5)
    show_plot = False
    show_loss = True
    train_split = 0.8
    neurons = 128
    result_dir = "training_results_3"
    num_epoch = 800
    file_paths = {i: f"../data/{i} sets.xlsx" for i in [114]}
    lrs = [0.001, 0.005]
    model_types = ["mlp_skip"]
    batch_sizes = [8, 16, 32]
    use_regs = [0., 0.05, 0.1, 0.5, 0.75]

    # Show the training progress of the neural network model
    a = time.time()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    pbar = tqdm(total=len(file_paths) * len(lrs) * len(model_types) * len(batch_sizes) * len(use_regs) * len(seeds))

    # Create a folder and read the file path
    for folder_name, file_path in file_paths.items():
        if not os.path.exists(result_dir):
            os.makedirs(f"{result_dir}/{folder_name}")
        data, _ = load_xlsx(file_path)
        # train_set, test_set = random_pair_split(whole_set, train_split)

        # The for loop is used to start the training of the neural network model
        for batch_size in batch_sizes:
            for lr in lrs:
                for model_type in model_types:
                    for use_reg in use_regs:
                        for seed in seeds:
                            np.random.seed(1)
                            train_set, test_set = random_pair_split(data, [train_split, 1 - train_split])
                            # Save the file path name
                            save_dir = f'{result_dir}/{folder_name}/{model_type}-sel{use_reg}' \
                                       f'-epoch{num_epoch}-lr{lr}-b{batch_size}-train{train_split}-seed{seed}/'
                            pbar.desc = "working on " + save_dir

                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                                os.mkdir(os.path.join(save_dir, 'img'))
                                os.mkdir(os.path.join(save_dir, 'model'))
                            if not os.path.exists(os.path.join(save_dir, "pred_train-test.xlsx")):
                                training_routine(train_set, test_set, model_type,
                                                 save_dir, device,
                                                 use_reg, batch_size, lr, neurons=neurons, seed=seed,
                                                 num_epoch=num_epoch, show_plot=show_plot, show_loss=show_loss
                                                 )
                            else:
                                print(f"skip {save_dir}")
                            pbar.update(1)
