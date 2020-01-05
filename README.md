# Graph-based-Recommendation-System
Graph convolutional matrix completion

## RUN
!python3 run.py --train_flag 1 \
                --test_flag 1 \
                --rate_num 5 \
                --use_side_feature 0 \
                --lr 0.01 \
                --weight_decay 0.00001 \
                --num_epochs 1000 \
                --hidden_dim 5 \
                --side_hidden_dim 5 \
                --out_dim 5 \
                --drop_out 0.0 \
                --split_ratio 0.8 \
                --save_steps 100 \
                --verbal_steps 100 \
                --log_dir './log' \
                --saved_model_folder './parameters' \
                --use_data_whitening 0 

1: 整理代码，数据和训练分隔开，所有function需要有注释
2：README，必要的安装包
3：tutorial notebook

# Higgs Boson Machine Learning Challenge
	.
	|-- README.md (here)
	|-- data (Please put train.csv and test.csv here)
	|-- src
	|   -- correction_rate.py
	|   -- costs.py
	|   -- evaluation.py
	|   -- feature_selection.py
	|   -- gradient_optimization.py
	|   -- implementations.py
	|   -- preprocess.py
	|   -- proj1_helpers.py
	|   -- run.py
	|   -- run.ipynb
    |   -- test.ipynb
	|   -- exploring data analysis.ipynb


# Recommender System

## Contents
- README.MD (this)
- submit

- data

- src


## Submission
To run run.py:
- On Mac/Windows : 
    - Open the Terminal, enter the zipped folder, enter to the folder **./src/**; 
    - To execute in  **./src/**, enter : python run.py;
    - submission.py is generated in *../submit/*;
    - item_feats_SGD.npy and user_feats_SGD.npy are stored in **../data/**.

## Codes
### Prerequisites
- Python 3.6+
- Numpy
- Scipy
- Pandas

> #### Note for running implement_surprise.py, some other libraries are required. Please refer to the script for details.

### Introduction
1. Notebook : 
    - Recommender_MF.ipynb: recorded how we analyzed the user-item ratings matrix and how we implemented the regularized MF and the biased MF. The notebook is organized as follows:

        1. Load data, split the ratings matrix into training and testing set
        2. Statistics analysis
        3. Presentation of the MF methods used
        4. Grid search of the best parameters (This part has been reorganized in gs_reg_MF.py and gs_biased_MF.py)
        5. Compute the predictions
        6. Creation of csv file for the submission.

2. Python modules :
    - data_process.py : This module transforms the data of a csv file into a sparse ratings matrix, split the data and functions to convert the final made predictions the correct format. 

    - SGD_helpers.py : This module initialize the parameters for matrix factorization, compute RMSE and compute the SGD.

    - MF_helpers.py : This module computes the bias of users and items and computes the global average.
