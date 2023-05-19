# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: config file
"""


save_ML_model = False              # Save ML models to 'models/'
get_lvl1_training_results = True   # this variable controls whether to output the training results of level_1 regressors
get_lvl2_training_results = True   # this variable controls whether to output the training results of level_2 regressors
get_statistical_model_summary = False # this variable controls whether to output the statistical (GEE, MixedLM) model summery
dataset_path = 'transformation_ds' # Save ML models to 'models/'
output_path = 'log/'                   # output path

# Exp_options, selects which regression model(s) to be performed
Exp_options = \
    {
        'run_SK_models' : 1,
        'run_H2O_models' : 1,
        'run_Statistical_models' : 0,
        'run_LSTM_model' : 0
    }