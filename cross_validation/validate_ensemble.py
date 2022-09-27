import utils
from train_model.train_torch import random_pair_split
from utils import preprocessing
import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, ARDRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

def grid_search_cv_ensemble(train_val_data, test_data, model_type, k_fold, seed):
    """
    :param train_val_data:training dataset
    :param test_data: testing dataset
    :param model_type: model type
    :param k_fold: The number of k-fold
    :param seed: random seeds
    :return: The MSE and R2 of each model are returned to obtain the optimal model and parameter Settings
    """
    def single_cv():

        np.random.seed(seed)
        train_val_data_ = preprocessing(train_val_data)
        test_data_ = preprocessing(test_data)

        if model_type == 'RF':
            parameters = {'estimator__n_estimators': [10, 50, 100, 200]}
            model = MultiOutputRegressor(RandomForestRegressor())
        elif model_type == 'Linear':
            parameters = {"estimator__fit_intercept": [True, False]}
            model = MultiOutputRegressor(LinearRegression())
        elif model_type == 'Ridge':
            parameters = {'estimator__alpha': [0.1, 0.5, 1.]}
            model = MultiOutputRegressor(Ridge())
        elif model_type == 'ARD':
            parameters = {'estimator__alpha_1': [1e-6, 5e-6, 1e-5],
                          'estimator__alpha_2': [1e-6, 5e-6, 1e-5],
                          'estimator__lambda_1': [1e-6, 5e-6, 1e-5],
                          'estimator__lambda_2': [1e-6, 5e-6, 1e-5]}
            model = MultiOutputRegressor(ARDRegression())
        elif model_type == "GBT":
            parameters = {'estimator__learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]}
            model = MultiOutputRegressor(GradientBoostingRegressor())
        elif model_type == "MLP":
            parameters = {
                'estimator__learning_rate_init': [0.005, 0.001, 0.0005],
                # 'estimator__tol': [1e-6],
                'estimator__batch_size': [8, 16, 32, 64],
                'estimator__hidden_layer_sizes': [32, 64, 128],
            }
            model = MultiOutputRegressor(MLPRegressor(activation='logistic'))
        elif model_type == "SVR":
            parameters = {'estimator__kernel': ('linear', 'rbf'), 'estimator__C': range(1, 11)}
            model = MultiOutputRegressor(SVR())
        elif model_type == "XGBoost":
            import xgboost as xgb
            parameters = {'estimator__booster': ("gbtree", "gblinear", "dart"),
                          'estimator__eta': [0.1, 0.2, 0.3, 0.4, 0.5]}
            model = MultiOutputRegressor(xgb.XGBRegressor())
        else:
            raise ValueError('Unexpected model type')

        clf = GridSearchCV(model, parameters, scoring="neg_mean_squared_error",
                           n_jobs=-1, refit=True, cv=k_fold, verbose=1, return_train_score=True)
        clf.fit(train_val_data_[:, :4], train_val_data_[:, 4:9])
        best_model = clf.best_estimator_
        pred_test = best_model.predict(test_data_[:, :4])
        pred_train = best_model.predict(train_val_data_[:, :4])
        test_MSE = mean_squared_error(test_data_[:, 4:9], pred_test)
        train_MSE = mean_squared_error(train_val_data_[:, 4:9], pred_train)
        test_R2 = r2_score(test_data_[:, 4:9], pred_test)
        train_R2 = r2_score(train_val_data_[:, 4:9], pred_train)

        rej_NaSO4, rej_MgCl2, rej_NaCl, rej_PEG = pred_test[:, 0], pred_test[:, 1], pred_test[:, 2], pred_test[:, 3]
        sel1 = np.log((1 - rej_NaCl) / (1 - rej_NaSO4))
        sel2 = np.log((1 - rej_NaCl) / (1 - rej_MgCl2))
        sel3 = np.log((1 - rej_NaCl) / (1 - rej_PEG))
        sel1[np.isnan(sel1)] = 0
        sel2[np.isnan(sel2)] = 0
        sel3[np.isnan(sel3)] = 0
        r2_sel1 = r2_score(sel1, test_data_[:, 9])
        r2_sel2 = r2_score(sel2, test_data_[:, 10])
        r2_sel3 = r2_score(sel3, test_data_[:, 11])
        MSE_sel1 = mean_squared_error(sel1, test_data_[:, 9])
        MSE_sel2 = mean_squared_error(sel2, test_data_[:, 10])
        MSE_sel3 = mean_squared_error(sel3, test_data_[:, 11])
        result = pd.DataFrame.from_dict({"CV_train_MSE": [-clf.cv_results_["mean_train_score"][clf.best_index_]],
                                         "CV_val_MSE": -clf.cv_results_["mean_test_score"][clf.best_index_],
                                         "train_MSE": [train_MSE],
                                         "train_R2": [train_R2],
                                         "test_MSE": [test_MSE],
                                         "test_R2": [test_R2],
                                         "test_sel_MSE": [(MSE_sel1 + MSE_sel2 + MSE_sel3) / 3],
                                         "test_sel_R2": [(r2_sel1 + r2_sel2 + r2_sel3) / 3],
                                         "best_param": [str(clf.best_params_)],
                                         "model_type": [model_type]})

        return result

    np.random.seed(seed)
    df = single_cv()
    return df




def random_search_cv_ensemble(train_val_data, test_data, model_type, k_fold, seed):
    """
    :param train_val_data:training dataset
    :param test_data: testing dataset
    :param model_type: model type
    :param k_fold: The number of k-fold
    :param seed: random seeds
    :return: The MSE and R2 of each model are returned to obtain the optimal model and parameter Settings
    """
    def single_cv():

        np.random.seed(seed)
        train_val_data_ = preprocessing(train_val_data)
        test_data_ = preprocessing(test_data)

        if model_type == 'RF':
            parameters = {"estimator__max_depth": [3, None],
                          "estimator__max_features": sp_randint(1, 11),
                          "estimator__min_samples_split": sp_randint(2, 11),
                          "estimator__min_samples_leaf": sp_randint(1, 11),
                          "estimator__bootstrap": [True, False],
                          # "estimator__criterion": ["gini", "entropy"],
                          'estimator__n_estimators': sp_randint(10, 200)}
            model = MultiOutputRegressor(RandomForestRegressor())
        elif model_type == 'Linear':
            parameters = {"estimator__fit_intercept": [True, False]}
            model = MultiOutputRegressor(LinearRegression())
        elif model_type == 'Ridge':
            parameters = {'estimator__alpha': uniform(0.1, 1.)}
            model = MultiOutputRegressor(Ridge())
        elif model_type == 'ARD':
            parameters = {'estimator__alpha_1': uniform(1e-6, 1e-4),
                          'estimator__alpha_2': uniform(1e-6, 1e-4),
                          'estimator__lambda_1': uniform(1e-6, 1e-4),
                          'estimator__lambda_2': uniform(1e-6, 1e-4)}
            model = MultiOutputRegressor(ARDRegression())
        elif model_type == "GBT":
            parameters = {'estimator__learning_rate': uniform(0.1, 0.5)}
            model = MultiOutputRegressor(GradientBoostingRegressor())
        elif model_type == "MLP":
            parameters = {
                'estimator__learning_rate_init': uniform(0.0005, 0.005),
                # 'estimator__tol': [1e-6],
                'estimator__batch_size': sp_randint(4, 32),
                'estimator__hidden_layer_sizes': sp_randint(32, 128),
            }
            model = MultiOutputRegressor(MLPRegressor(activation='logistic'))
        elif model_type == "SVR":
            parameters = {'estimator__kernel': ('linear', 'rbf'), 'estimator__C': range(1, 11)}
            model = MultiOutputRegressor(SVR())
        elif model_type == "XGBoost":
            import xgboost as xgb
            parameters = {'estimator__booster': ("gbtree", "gblinear", "dart"),
                          'estimator__eta': uniform(0.1, 0.5)}
            model = MultiOutputRegressor(xgb.XGBRegressor())
        else:
            raise ValueError('Unexpected model type')

        clf = RandomizedSearchCV(model, parameters, scoring="neg_mean_squared_error", n_iter=144,
                           n_jobs=-1, refit=True, cv=k_fold, verbose=1, return_train_score=True)
        clf.fit(train_val_data_[:, :4], train_val_data_[:, 4:9])
        best_model = clf.best_estimator_
        pred_test = best_model.predict(test_data_[:, :4])
        pred_train = best_model.predict(train_val_data_[:, :4])
        test_MSE = mean_squared_error(test_data_[:, 4:9], pred_test)
        train_MSE = mean_squared_error(train_val_data_[:, 4:9], pred_train)
        test_R2 = r2_score(test_data_[:, 4:9], pred_test)
        train_R2 = r2_score(train_val_data_[:, 4:9], pred_train)

        rej_NaSO4, rej_MgCl2, rej_NaCl, rej_PEG = pred_test[:, 0], pred_test[:, 1], pred_test[:, 2], pred_test[:, 3]
        sel1 = np.log((1 - rej_NaCl) / (1 - rej_NaSO4))
        sel2 = np.log((1 - rej_NaCl) / (1 - rej_MgCl2))
        sel3 = np.log((1 - rej_NaCl) / (1 - rej_PEG))
        sel1[np.isnan(sel1)] = 0
        sel2[np.isnan(sel2)] = 0
        sel3[np.isnan(sel3)] = 0
        r2_sel1 = r2_score(sel1, test_data_[:, 9])
        r2_sel2 = r2_score(sel2, test_data_[:, 10])
        r2_sel3 = r2_score(sel3, test_data_[:, 11])
        MSE_sel1 = mean_squared_error(sel1, test_data_[:, 9])
        MSE_sel2 = mean_squared_error(sel2, test_data_[:, 10])
        MSE_sel3 = mean_squared_error(sel3, test_data_[:, 11])
        result = pd.DataFrame.from_dict({"CV_train_MSE": [-clf.cv_results_["mean_train_score"][clf.best_index_]],
                                         "CV_val_MSE": -clf.cv_results_["mean_test_score"][clf.best_index_],
                                         "train_MSE": [train_MSE],
                                         "train_R2": [train_R2],
                                         "test_MSE": [test_MSE],
                                         "test_R2": [test_R2],
                                         "test_sel_MSE": [(MSE_sel1 + MSE_sel2 + MSE_sel3) / 3],
                                         "test_sel_R2": [(r2_sel1 + r2_sel2 + r2_sel3) / 3],
                                         "best_param": [str(clf.best_params_)],
                                         "model_type": [model_type]})

        return result

    np.random.seed(seed)
    df = single_cv()
    return df




def bayesian_search_cv_ensemble(train_val_data, test_data, model_type, k_fold, seed):
    """
    :param train_val_data:training dataset
    :param test_data: testing dataset
    :param model_type: model type
    :param k_fold: The number of k-fold
    :param seed: random seeds
    :return: The MSE and R2 of each model are returned to obtain the optimal model and parameter Settings
    """
    def single_cv():

        np.random.seed(seed)
        train_val_data_ = preprocessing(train_val_data)
        test_data_ = preprocessing(test_data)

        if model_type == 'RF':
            parameters = {"estimator__max_depth": Categorical([3, None]),
                          "estimator__max_features": Integer(1, 11),
                          "estimator__min_samples_split": Integer(2, 11),
                          "estimator__min_samples_leaf": Integer(1, 11),
                          "estimator__bootstrap": [True, False],
                          # "estimator__criterion": ["gini", "entropy"],
                          'estimator__n_estimators': Integer(10, 200)}
            model = MultiOutputRegressor(RandomForestRegressor())
        elif model_type == 'Linear':
            parameters = {"estimator__fit_intercept": [True, False]}
            model = MultiOutputRegressor(LinearRegression())
        elif model_type == 'Ridge':
            parameters = {'estimator__alpha': Real(0.1, 1.)}
            model = MultiOutputRegressor(Ridge())
        elif model_type == 'ARD':
            parameters = {'estimator__alpha_1': Real(1e-6, 1e-4),
                          'estimator__alpha_2': Real(1e-6, 1e-4),
                          'estimator__lambda_1': Real(1e-6, 1e-4),
                          'estimator__lambda_2': Real(1e-6, 1e-4)}
            model = MultiOutputRegressor(ARDRegression())
        elif model_type == "GBT":
            parameters = {'estimator__learning_rate': Real(0.1, 0.5)}
            model = MultiOutputRegressor(GradientBoostingRegressor())
        elif model_type == "MLP":
            parameters = {
                'estimator__learning_rate_init': Real(0.0005, 0.005),
                # 'estimator__tol': [1e-6],
                'estimator__batch_size': Integer(4, 32),
                'estimator__hidden_layer_sizes': Integer(32, 128),
            }
            model = MultiOutputRegressor(MLPRegressor(activation='logistic'))
        elif model_type == "SVR":
            parameters = {'estimator__kernel': Categorical(['linear', 'rbf']), 'estimator__C': Integer(1, 11)}
            model = MultiOutputRegressor(SVR())
        elif model_type == "XGBoost":
            import xgboost as xgb
            parameters = {'estimator__booster': Categorical(["gbtree", "gblinear", "dart"]),
                          'estimator__eta': Real(0.1, 0.5)}
            model = MultiOutputRegressor(xgb.XGBRegressor())
        else:
            raise ValueError('Unexpected model type')

        clf = BayesSearchCV(model, parameters, scoring="neg_mean_squared_error", n_iter=144, n_points=20,
                           n_jobs=-1, refit=True, cv=k_fold, verbose=1, return_train_score=True)
        clf.fit(train_val_data_[:, :4], train_val_data_[:, 4:9])
        best_model = clf.best_estimator_
        pred_test = best_model.predict(test_data_[:, :4])
        pred_train = best_model.predict(train_val_data_[:, :4])
        test_MSE = mean_squared_error(test_data_[:, 4:9], pred_test)
        train_MSE = mean_squared_error(train_val_data_[:, 4:9], pred_train)
        test_R2 = r2_score(test_data_[:, 4:9], pred_test)
        train_R2 = r2_score(train_val_data_[:, 4:9], pred_train)

        rej_NaSO4, rej_MgCl2, rej_NaCl, rej_PEG = pred_test[:, 0], pred_test[:, 1], pred_test[:, 2], pred_test[:, 3]
        sel1 = np.log((1 - rej_NaCl) / (1 - rej_NaSO4))
        sel2 = np.log((1 - rej_NaCl) / (1 - rej_MgCl2))
        sel3 = np.log((1 - rej_NaCl) / (1 - rej_PEG))
        sel1[np.isnan(sel1)] = 0
        sel2[np.isnan(sel2)] = 0
        sel3[np.isnan(sel3)] = 0
        r2_sel1 = r2_score(sel1, test_data_[:, 9])
        r2_sel2 = r2_score(sel2, test_data_[:, 10])
        r2_sel3 = r2_score(sel3, test_data_[:, 11])
        MSE_sel1 = mean_squared_error(sel1, test_data_[:, 9])
        MSE_sel2 = mean_squared_error(sel2, test_data_[:, 10])
        MSE_sel3 = mean_squared_error(sel3, test_data_[:, 11])
        result = pd.DataFrame.from_dict({"CV_train_MSE": [-clf.cv_results_["mean_train_score"][clf.best_index_]],
                                         "CV_val_MSE": -clf.cv_results_["mean_test_score"][clf.best_index_],
                                         "train_MSE": [train_MSE],
                                         "train_R2": [train_R2],
                                         "test_MSE": [test_MSE],
                                         "test_R2": [test_R2],
                                         "test_sel_MSE": [(MSE_sel1 + MSE_sel2 + MSE_sel3) / 3],
                                         "test_sel_R2": [(r2_sel1 + r2_sel2 + r2_sel3) / 3],
                                         "best_param": [str(clf.best_params_)],
                                         "model_type": [model_type]})

        return result

    np.random.seed(seed)
    df = single_cv()
    return df
if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    k = 5
    seeds = [1]
    file_paths = {i: f"../data/{i} sets.xlsx" for i in [114]}
    model_types = ["MLP", "RF", "Linear", "Ridge", "ARD", "GBT", "SVR", "XGBoost"]
    search_method = "bayesian"
    # model_types = ["SVR"]

    for data_name, data_path in file_paths.items():
        results = []
        save_name = f"{search_method}-ensemble_baseline5-{data_name}.xlsx"
        data, column_names = utils.load_xlsx(data_path)

        writer = pd.ExcelWriter(save_name, engine='openpyxl')
        all_df = []
        for model_type in model_types:
            for seed in seeds:
                print("----------------operating " + model_type + '------------------')
                np.random.seed(3)
                train_val_data, testing_data = random_pair_split(data, [0.8, 0.2])
                if search_method == "random":
                    df = random_search_cv_ensemble(train_val_data, testing_data, model_type, k, seed)
                elif search_method == "grid":
                    df = grid_search_cv_ensemble(train_val_data, testing_data, model_type, k, seed)
                elif search_method == "bayesian":
                    df = bayesian_search_cv_ensemble(train_val_data, testing_data, model_type, k, seed)
                else:
                    raise NotImplementedError
                df["seed"] = seed
                df["model_type"] = model_type
                all_df.append(df)

        df_summary = pd.concat(all_df)
        df_summary["train_RMSE"] = 100 * df_summary["train_MSE"].apply(lambda x: np.sqrt(x))
        # df_summary["val_RMSE"] = 100 * df_summary["val_MSE"].apply(lambda x: np.sqrt(x))
        df_summary["test_RMSE"] = 100 * df_summary["test_MSE"].apply(lambda x: np.sqrt(x))
        df_summary.to_excel(writer, sheet_name="summary", startrow=0, index=False)
        df_summary.to_excel(writer, startrow=0, index=False)
        writer.save()
