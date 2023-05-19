# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: Adjusted_R2, R2, MAE, RMSE, and MAPE scoring methods
"""

import numpy as np
from ._write_to_file import write_to_file
from ._time_utils import get_timestamp

######################################################################
# SKlearn, H2O, and statistical models' metrics
###############################################
from sklearn.metrics import mean_absolute_error as MAE_sklearn,\
                            mean_squared_error as MSE_sklearn,\
                            r2_score as R2_score_sklearn
from sklearn.metrics.regression import _check_reg_targets

def RMSE(y_true, y_pred):
    return np.sqrt(MSE_sklearn(y_true, y_pred))

def MAPE(y_true, y_pred, multioutput='uniform_average'):
    '''
    This function is implemented by: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>, 
									 the implementation of MAPE function follows sklearn/metrics regression metrics
    Parameters
    ----------
        y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Ground truth (correct) target values.

        y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Estimated target values.
    '''
    _, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, multioutput)
    assert not(0.0 in y_true), 'MAPE arrises an Error, cannot calculate MAPE while y_true has 0 element(s). Check \"utils\_scoring_metrics.MAPE"'
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_score_________(df, model_name, data_set_name, get_str = 0, x_shape=None, timestamp=False):
    # x_shape is used to calculate Adj-R2
    y_true = df['y_true_shifted']
    y_pred = df['y_pred']
    if x_shape:s = '' + str(model_name) + get_score_string(y_true = y_true, y_pred = y_pred, x_shape=x_shape, timestamp = timestamp)
    else:s = '' + str(model_name) + get_score_string(y_true = y_true, y_pred = y_pred, timestamp = timestamp)
    if get_str:return s + '  '
    else: write_to_file(data_set_name, s)

def get_score(df, model_name, y_pred, n, data_set_name, get_str = 0, x_shape=None):
    # x_shape is used to calculate Adj-R2
    if n:
        y_true = df['price_scaled_'+str(n)]
    else:
        y_true = df['price_scaled']

    if not n:s = '' + str(model_name) + get_score_string(y_true = y_true, y_pred = y_pred, x_shape=x_shape)
    else:s = '' + str(model_name) + get_score_string(y_true = y_true, y_pred = y_pred)
    if get_str:return s + '  '
    else: write_to_file(data_set_name, s)

def get_score_string(y_true, y_pred, x_shape=None, timestamp=False):
    # x_shape is used to calculate Adj-R2
    r_squared = R2_score_sklearn(y_true = y_true,y_pred = y_pred)
    s = 'R2:'+ str(r_squared*100)[:6]+\
        '  MAE:' + str(MAE_sklearn(y_true,y_pred) )[:6] +\
        '  RMSE:' + str(RMSE(y_true,y_pred) )[:6] +\
        '  MAPE:' + str(MAPE(y_true,y_pred) )[:6]
    if x_shape: s+= '  Adj_R2:' + str((1 - (1-r_squared)*(x_shape[0]-1)/(x_shape[0]-x_shape[1]-1))*100)[:6]
    else: s+= ' '*15
    # return s + ' '*10 + get_timestamp(1)
    return s + ' '*10 + get_timestamp(1) if timestamp else s

def get_h2o_score_string(model_performance, MAPE_, n = 100 , x_shape = None, timestamp=False):
    s = 'R2:%-6s  MAE:%-6s  RMSE:%-6s  MAPE:%-6s'%(str(model_performance.r2()*100)[:6],
                                                      str(model_performance.mae())[:6],
                                                      str(model_performance.rmse())[:6],
                                                      str(MAPE_)[:6])
    if not n: s+= '  Adj_R2:' + str((1 - (1-model_performance.r2())*(x_shape[0]-1)/(x_shape[0]-x_shape[1]-1))*100)[:6]
    else: s+= ' '*15
    return s + ' '*10 + get_timestamp(1) if timestamp else s
    # return s + ' '*10 + get_timestamp(1)

def get_h2o_score(data_set_name, model_name, model_performance, get_str = 0):
    if '^' in model_name:
        model_name = model_name.replace('^','\n')
    if get_str:return str(model_name)+model_performance + '  '
    else:write_to_file(data_set_name, "  ",str(model_name),model_performance)


###################################
# Statistical models function
####################
def get_statistical_model_results(model, model_name, x_train, x_test, feature_str, fitting_time, data_set_name, get_statistical_model_summary):
	
	# training Score
    y_pred = model.predict(exog=x_train[[i.strip() for i in feature_str.split(' + ')]])
    write_to_file(data_set_name,'    Train %-15s'%model_name,get_score_string(x_train["y"], y_pred))
	
	# testing Score
    y_pred = model.predict(exog=x_test[[i.strip() for i in feature_str.split(' + ')]])
    write_to_file(data_set_name,'    %s %-15s'%( fitting_time, model_name), get_score_string(x_test["y"], y_pred), '\n')
	
    if get_statistical_model_summary:
        write_to_file(data_set_name,model.summary(),'\n'*10,'#'*80)

###################################
# LSTM Keras metrics
####################
from keras import backend as K
def R2_keras(y_true, y_pred):
    # kaggle: https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34019
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def MAE_keras(y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true), axis=-1)
def RMSE_keras(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def MAPE_keras(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),K.epsilon(),None))
    return 100. * K.mean(diff, axis=-1)

def get_LSTM_score(df, X, model, model_name_str, n, number_of_categories):
    y_pred = get_LSTM_prediction(X, model, model_name_str, number_of_categories)
    if n:
        y_true = df['price_scaled_'+str(n)]
    else:
        y_true = df['price_scaled']

    return get_score_string(y_true = y_true, y_pred = y_pred)

def get_LSTM_prediction(X, model, model_name_str, number_of_categories):
    model.load_weights(model_name_str)
    return model.predict([X[:, :-number_of_categories ],X[:, -number_of_categories:]])


