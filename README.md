#  MembraneAI: Machine Learning Guided Polyamide Membrane with Exceptional Solute-Solute Selectivity and Permeance
This repo is a support for our publication **Machine Learning Guided Polyamide Membrane with Exceptional Solute-Solute Selectivity and Permeance**, developed by [Zhiyao Luo](https://github.com/GilesLuo). 
A special thanks goes to [Yaotian Yang](https://github.com/yyt-2378) for his generous help on sorting out this repo.

## General info
Herein, we present a data-driven strategy to guide the design of polyamide membrane by developing a novel artificial neural network (ANN) 
based multi-task machine learning (ML) model with skip connections and selectivity regularization. We used limited sets of laboratory 
collected data to obtain satisfactory model performance over four iterations of an ever-expanding dataset incorporating with human 
expert experience in the online learning process. We then fabricated four membranes under the guidance of model, and they all exceed 
the present upper bound for mono/divalent ion selectivity and permeance of the polymeric membranes. Moreover, we obtained mechanistic 
insights into the membrane design through feature analysis of the model. Our work demonstrates that ML approach represents a paradigm 
shift for high-performance polymeric membranes design. 

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Use](#use)
* [File Structure](#file-structure)
	
## Setup
To run the code, first ensure that the Python Interpreter version is `python3.8` or another appropriate version, and then install the 
appropriate module using the ```pip install -r requirements.txt``` .

## use
To simply run by the train-test manner, run `train_torch.py` directly<br>

Alternatively,  run `train_validate.py` to get the optimal hyperparameters on the train-val dataset, 
and retrain models by running `train_torch.py` with the optimized hyperparameters.
![image](https://github.com/GilesLuo/pred-membrane/blob/Image/image1.png)

## file structure
```
│ 
├─utils.py                  # helper functions for data split and training/testing
│
├─cross_validation          # Use k-fold test method of cross validation to evaluate the model performance and optimize model overparameters
│      validate_ensemble.py # Validate emsemble models such as random forest and SVR
│      validate_nn.py       # Validate ANN-based models such as native MLP and our model
│      __init__.py
│      
├─data
│       # datasets for 5 iterations, named by the number of unique samples
│      
├─train_model
│      torch_models.py       # Different MLP models are defined and written
│      train_torch.py        # The body function for training, which can be run directly
│      __init__.py
│      
└─visu                       # Visualization of data results and data characteristics
        draw_pdp.py
        eval_multiregression.py
        feature_importance.py
        plotting.py
        __init__.py
```


