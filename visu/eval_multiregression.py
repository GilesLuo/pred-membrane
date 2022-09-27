import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from draw_pdp import save_pdp_1D, save_pdp_2D
from train_model.train_torch import MembraneSet, DataLoader


# Generate predicted values from a given model
def generate_prediction(input1, input2, input3, input4, model, file_name):
    '''
    @param input1: input value
    @param input2: input value
    @param input3: input value
    @param input4: input value
    @param model: model
    @param file_name: save_file name
    @param sel_optimal: bool=False
    @return: Returns the predicted data set
    '''
    input_array = []
    for i1 in input1:
        for i2 in input2:
            for i3 in input3:
                for i4 in input4:
                    input_array.append([i1, i2, i3, i4])

    # Converts the input layer data into an array of the specified columns and reads the data in batches through the MPeset and DataLoader
    input_array = np.array(input_array).reshape(-1, 4)
    test_input_dataset = MembraneSet(input_array, gen=True)
    test_input_dataloader = DataLoader(test_input_dataset, batch_size=1200, shuffle=False)
    model.eval()
    results = []
    entry_counter = 0
    # Show the progress of data processing by specifying Model for data prediction
    pbar = tqdm(total=len(test_input_dataloader))
    for X, _ in test_input_dataloader:
        pred = model(X)

        # restrict maximum predicted rejection to 0.998 due to the limited
        # precision of experimental devices
        pred[:, 0][pred[:, 0] > 0.998] = 0.998
        pred[:, 1][pred[:, 1] > 0.998] = 0.998

        # revert input features
        X[:, 1] = np.exp(X[:, 1])
        X[:, 2] = X[:, 2] * 2.05
        X[:, 3] *= 60.

        selectivity1_cp = (1. - pred[:, 2]) / (1. - pred[:, 0])
        selectivity2_cp = (1. - pred[:, 2]) / (1. - pred[:, 1])
        selectivity3_cp = (1. - pred[:, 2]) / (1. - pred[:, 3])
        result = torch.cat([X,
                            pred,
                            selectivity1_cp.reshape(-1, 1),
                            selectivity2_cp.reshape(-1, 1),
                            selectivity3_cp.reshape(-1, 1),
                            ], dim=1)

        results.append(result)
        entry_counter += result.shape[0]
        pbar.desc = "length of file {}".format(entry_counter)
        pbar.update()
    results = torch.cat(results, dim=0).detach().numpy()

    # The data is written to the DataFrame
    import pandas as pd
    df = pd.DataFrame(results)
    df.columns = ['CPIP', 'CTMC', 'CSDS', 'ti', 'Na2SO4', 'MgCl2', 'NaCl', 'PEG400', 'PWP',
                  'sel1', 'sel2', 'sel3']
    df_to_return = df.copy()
    try:
        writer = pd.ExcelWriter(file_name, engine='openpyxl')

        df.to_excel(writer, startrow=0, index=False)
        writer.save()
    except:
        Warning("excel file too large, skip saving")
        pass
    return df_to_return


def eval_routine(input1, input2, input3, input4, save_dir, epoch, mode):
    '''
    @param input1: input value
    @param input2: input value
    @param input3: input value
    @param input4: input value
    @param save_dir: save_file path
    @param epoch: The number of complete training sessions
    @param mode: bool
    '''
    data, model = None, None
    # Save file path name generation
    pred_file_name = os.path.join(save_dir, f"prediction/pred-epoch{epoch}.xlsx")
    PD_1D_file_name = os.path.join(save_dir, f"prediction/pd_1D-epoch{epoch}.xlsx")
    PD_2D_file_name = os.path.join(save_dir, f"prediction/pd_2D-epoch{epoch}.xlsx")
    # with open(save_dir + "training_log.txt") as f:
    #     pass
    if not os.path.exists(os.path.join(save_dir, 'prediction/')):
        os.mkdir(os.path.join(save_dir, 'prediction/'))
    if not os.path.exists(pred_file_name):
        print('generating predictions...')
        model = torch.load(os.path.join(save_dir, 'model', f"epoch{epoch}.pth"))
        model.give_importance = True
        data = generate_prediction(input1, input2, input3, input4,
                                   model, pred_file_name)

    # The dataset is written to an XLSX file
    if not os.path.exists(PD_1D_file_name):
        print('computing 1D partial dependencies...')
        if data is None:
            data = pd.read_excel(pred_file_name, engine="openpyxl")
        save_pdp_1D(data, save_name=PD_1D_file_name, mode=mode)
    if not os.path.exists(PD_2D_file_name):
        if data is None:
            data = pd.read_excel(pred_file_name, engine="openpyxl")
        print('computing 2D partial dependencies...')
        save_pdp_2D(data, save_name=PD_2D_file_name, mode=mode)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # Enter each parameter in turn, and call a function to predict and visualize the result
    save_dir = 'training_results_2/100/mlp-sel0.8-epoch700-lr0.005-b8-train0.9999-fold0'
    epoch = 672
    mode = "relative"
    input1 = [0.1, 0.25, 0.5, 0.75, 1, 2]
    input2 = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.045,
              0.05, 0.1, 0.12, 0.15, 0.16, 0.2, 0.4, 0.6, 0.8, 1.6]
    input3 = [0, 2.05, 4.1, 8.2, 16.4, 24.6]
    input4 = np.array([30, 60, 120])
    eval_routine(input1, input2, input3, input4, save_dir,
                 epoch, mode)
