import torch
import warnings
# warnings.filterwarnings("ignore")
import utils
import os
import copy
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd
import numpy as np
from tqdm import tqdm
from shutil import copytree, ignore_patterns
from train_model.train_torch import random_pair_split, training_routine, MembraneSet, train_loop, test_loop, loss_fn
from train_model.torch_models import MLP_skip, MLP


def cross_valid_nn(train_val_data_, model_type, lr, use_reg, neuron,
                   k_fold, batch_size, num_epoch, device, seeds, save_name):
    '''
    :param train_val_data_: training dataset
    :param model_type: model type
    :param lr: learning rate
    :param use_reg: use_reg is a hyperparameter to set the proportion of regularization regarding the norm of main loss.
    :param neuron: Number of neurons
    :param k_fold: The data was divided into N pieces for cross validation
    :param batch_size: batch size
    :param num_epoch: The number of the epoch
    :param device: device
    :param seeds: random seeds
    :param save_name: The name of the saved file
    :return: final_df
    '''

    def single_cv(model):
        summary_name = os.path.join(save_name, f"seed{seed}", "summary.xlsx")
        if os.path.exists(summary_name):
            df = pd.read_excel(summary_name)
            print(f"found existing summary in {summary_name}, skip")
            return df
        else:
            print(f"generate {summary_name}")
            train_loss_score, val_loss_score, converge_epoch = [], [], []

            loss = loss_fn(use_reg)
            train_val_data_bin = random_pair_split(train_val_data, [1 / k_fold] * k_fold)
            datasets = [MembraneSet(d) for d in train_val_data_bin]
            for i in range(k_fold):
                save_name_ = os.path.join(save_name, f"seed{seed}")
                if not os.path.exists(save_name_):
                    os.makedirs(save_name_, exist_ok=True)
                datasets_copy = datasets[:]
                val_set = datasets_copy.pop(i)
                train_set = torch.utils.data.ConcatDataset(datasets_copy)
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                           shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_set, batch_size=500,
                                                         shuffle=True)

                tr_losses, val_losses = [], []
                for epoch in range(1, num_epoch + 1):
                    tr_l, _ = train_loop(train_loader, model, loss, optimizer, device)
                    val_l, _ = test_loop(val_loader, model, device)
                    tr_losses.append(np.round(tr_l, 5))
                    val_losses.append(np.round(val_l, 5))
                    # early stopping
                    if len(tr_losses) - np.argmin(tr_losses) > 50 and len(tr_losses) - np.argmin(val_losses) > 50:
                        break
                    scheduler.step()
                min_val_loss = np.min(val_losses)
                best_indices = np.where(val_losses == min_val_loss)[0]
                if len(best_indices) == 1:
                    best_idx = best_indices[0]
                else:
                    best_idx = best_indices[np.argmin(np.array(tr_losses)[best_indices])]
                converge_epoch.append(best_idx + 1)
                train_loss_score.append(tr_losses[best_idx])
                val_loss_score.append(val_losses[best_idx])
            result = np.array(
                [train_loss_score, val_loss_score, converge_epoch]).transpose()
            df = pd.DataFrame(result, columns=["train_MSE",
                                               "val_MSE",
                                               "converge_epoch"])
            df["fold"] = range(k_fold)
            df["seed"] = seed
            writer_single = pd.ExcelWriter(summary_name,
                                           engine='openpyxl')
            df.to_excel(writer_single, startrow=0, index=False)
            writer_single.save()
            writer_single.close()
            return df

    if not os.path.exists(save_name):
        os.mkdir(save_name)
    all_df = []
    train_val_data = copy.deepcopy(train_val_data_)
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if model_type == "mlp":
            model = MLP(4, 5, neurons=neuron).to(device)
        elif model_type == "mlp_skip":
            model = MLP_skip(4, 5, neurons=neuron).to(device)
        else:
            raise ValueError
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.95)
        df = single_cv(model)
        all_df.append(df)
    final_df = pd.concat(all_df, ignore_index=True)
    writer1 = pd.ExcelWriter(save_name + ".xlsx", engine='openpyxl')
    final_df.to_excel(writer1, startrow=0, index=False)
    writer1.save()
    writer1.close()
    return final_df


def mp_validate_nn(data, model_types, batch_sizes, use_regs, neurons,
                   lr, k, seeds, save_dir, save_name, num_epoch=500,
                   num_core=8, device="cpu"):
    '''
    @param data: dataset
    @param model_types: model type
    @param batch_sizes: batch size
    @param use_regs: is a hyperparameter to set the proportion of regularization regarding the norm of main loss.
    @param neurons: Number of neurons
    @param lr: learning rate
    @param k: The value of k in the k-fold
    @param seeds: seeds
    @param save_dir: Save the file path name
    @param save_name: Save the name
    @param num_epoch: The number of the epoch
    @param num_core: Number of cpus in use
    @param device: device
    @return: test results
    '''
    jobsCompleted = 0
    counter = 0
    parallel_pool = Pool(num_core)
    names = []
    configs = []
    processes = []
    np.random.seed(3)
    # Divide the dataset in a given proportion
    train_val_data, testing_data = random_pair_split(data, [0.8, 0.2])
    # Call the model for data training, and use multi-thread operation at the same time, improve the utilization of CPU
    for num_neuron in neurons:
        for batch_size in batch_sizes:
            for model_type in model_types:
                for use_reg in use_regs:
                    for lr_ in lr:
                        configs.append([model_type, use_reg, batch_size, num_neuron, lr_])
                        name = f"{save_dir}/{model_type}-reg{use_reg}-bs{batch_size}-neu{num_neuron}-lr{lr_}"
                        names.append(name)
                        if not os.path.exists(name + '.xlsx'):
                            if num_core > 1:
                                p = parallel_pool.apipe(cross_valid_nn,*(
                                    train_val_data, model_type, lr_, use_reg,
                                    num_neuron, k, batch_size,
                                    num_epoch,
                                    device, seeds, name))

                                processes.append([p, counter])
                                counter += 1
                            else:
                                cross_valid_nn(train_val_data, model_type, lr_, use_reg, num_neuron, k,
                                               batch_size, num_epoch,
                                               device, seeds, name)
    # Shows the progress of training
    pbar = tqdm(total=len(processes))
    while len(processes) > 0:
        for i in range(len(processes)):
            task, j = processes[i]
            if task.ready():
                jobsCompleted += 1
                pbar.desc = names[j]
                pbar.update()
                task.get()
                processes.pop(i)
                break
    pbar.close()

    # Write result to the XLSX file
    results = []
    for name, config in tqdm(zip(names, configs), desc='get validation result'):
        result = pd.read_excel(name + '.xlsx', engine="openpyxl")
        result['model_type'], result['reg'], result['batch_size'], result['num_neuron'], result['lr'] = config
        # result["name"] = os.path.join(name, f"seed{result['seed']}", f"fold{result['fold']}")
        results.append(result)
    results = pd.concat(results)
    results["train_RMSE"] = 100 * results["train_MSE"].apply(lambda x: np.sqrt(x))
    results["val_RMSE"] = 100 * results["val_MSE"].apply(lambda x: np.sqrt(x))
    results.set_index(['model_type', 'reg', 'batch_size', 'num_neuron', 'lr', "seed", "fold"], inplace=True)
    results.sort_values('val_MSE', ascending=True, inplace=True)
    summary_fold_seed = results.groupby(['model_type', 'reg', 'batch_size', 'num_neuron', 'lr']).mean()

    best_hparam = summary_fold_seed.groupby(['model_type', 'reg'])["val_MSE"].idxmin()
    summary_all = summary_fold_seed.loc[best_hparam]

    writer = pd.ExcelWriter(save_name, engine='openpyxl')
    results.to_excel(writer, sheet_name="all", startrow=0, index=True)
    summary_fold_seed.to_excel(writer, sheet_name="summary_fold_seed", startrow=0, index=True)
    summary_all.to_excel(writer, sheet_name="summary_all", startrow=0, index=True)
    writer.save()
    writer.close()

    # Write test_results to the XLSX file
    test_results = []
    pbar = tqdm(total=summary_all.shape[0] * 5, desc=f"testing for {folder_name}")
    for row in summary_all.iterrows():
        model_type, use_reg, batch_size, num_neuron, lr = row[0]
        for seed in range(1, 5):
            save_dir = f'CV_test_{folder_name}/{model_type}-sel{use_reg}' \
                       f'-epoch{num_epoch}-lr{lr}-b{batch_size}-seed{seed}/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                os.mkdir(os.path.join(save_dir, 'img'))
                os.mkdir(os.path.join(save_dir, 'model'))
            if len(os.listdir(os.path.join(save_dir, "model"))) != num_epoch:
                df = training_routine(train_val_data, testing_data, model_type=model_type,
                                      save_dir=save_dir, device=device,
                                      use_reg=use_reg, batch_size=batch_size, lr=lr, seed=seed, neurons=num_neuron,
                                      num_epoch=num_epoch, show_plot=False, show_loss=False, save_models=False
                                      )
            else:
                df = pd.read_excel(os.path.join(save_dir, "training_log.xlsx"), engine="openpyxl")
            best_loss = df.loc[df["test_sel_MSE"].idxmin()]
            best_loss = pd.concat(
                [best_loss, pd.Series(row[0], ['model_type', 'use_reg', 'batch_size', 'num_neuron', 'lr'])])
            best_loss["seed"] = seed
            test_results.append(best_loss)
            pbar.update()
    test_results = pd.concat(test_results, axis=1).T
    test_results.set_index(['model_type', 'use_reg', 'batch_size', 'num_neuron', 'lr', "seed"], inplace=True)

    test_summary = test_results.groupby(['model_type', 'use_reg', 'batch_size', 'num_neuron', 'lr']).mean()

    test_summary["train_rej_RMSE"] = 100 * test_summary["train_MSE"].apply(lambda x: np.sqrt(x))
    test_summary["test_rej_RMSE"] = 100 * test_summary["test_MSE"].apply(lambda x: np.sqrt(x))
    test_summary["train_sel_RMSE"] = test_summary["train_sel_MSE"].apply(lambda x: np.sqrt(x))
    test_summary["test_sel_RMSE"] = test_summary["test_sel_MSE"].apply(lambda x: np.sqrt(x))

    test_results.sort_values("test_MSE", ascending=True)
    writer = pd.ExcelWriter(f'nn_CV_test_{folder_name}.xlsx', engine='openpyxl')
    test_results.to_excel(writer, sheet_name="test_all", startrow=0, index=True)
    test_summary.to_excel(writer, sheet_name="test_summary", startrow=0, index=True)
    writer.save()
    writer.close()
    return test_results


if __name__ == "__main__":
    # Assign the hyperparameters in turn, train the model with the MP_validate_nn function, and return test_results
    k = 5
    seeds = [1]
    model_types = ['mlp', 'mlp_skip']
    batch_sizes = [8, 16, 32, 64]
    use_regs = [0., 0.25, 0.5, 0.75]
    neurons = [32, 64, 128]
    lr = [0.005, 0.001, 0.0005]
    device = "cpu"
    num_epoch = 600
    num_core = 6

    file_paths = {i: f"../data/{i} sets.xlsx" for i in [114, 100, 84, 71, 57]}
    for folder_name, file_path in file_paths.items():
        save_dir = f'CV_{folder_name}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            os.mkdir(f'CV_test_{folder_name}')
        save_name = f"nn_CV_baseline{folder_name}.xlsx"
        data, _ = utils.load_xlsx(file_path)
        results = mp_validate_nn(data, model_types, batch_sizes, use_regs, neurons,
                                 lr, k, seeds, save_dir, save_name, num_epoch, num_core, device)
