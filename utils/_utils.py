# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: util functions
"""

import numpy as np
import pandas as pd

from ._transformation import scalling_shifted_prices, log_scaling_transformation, get_mean_values, transformation_coef_lst, Transform, apply_transformation
from ._scoring_metrics import get_score_string
from ._shifting_avgerage import apply_shiftting, Back_shift
from ._write_to_file import write_to_file

def get_train_test_indexes(df):
    from sklearn.model_selection import train_test_split
    train,test = train_test_split( list(range(df.shape[0])) , test_size=0.2, random_state=1, shuffle=True, stratify=df.values )
    return train, test

def train_test_split_(X_df, y_df, train_index, test_index):
    if not X_df is None:
        if type(X_df) is pd.core.frame.DataFrame:
            X_train, X_test  = X_df.iloc[train_index], X_df.iloc[test_index]
        else:
            X_train, X_test  = X_df[train_index], X_df[test_index]
    else:
        X_train, X_test  = None, None

    if not y_df is None:
        y_train, y_test = y_df.iloc[train_index], y_df.iloc[test_index]
    else:
        y_train, y_test  = None, None

    return X_train, y_train, X_test, y_test

def get_Exp_str(str_, model_name, df_len):
    from ._time_utils import get_timestamp
    return str_ + ' %-15s '%model_name + 'len_%-8d '%(df_len) + get_timestamp()

def handle_df(df, outlier_type, data_set_name, scaling_lvl2 = True, print_samples = 0):
    if outlier_type == 'IQR':
        for cou,i in enumerate(df.Categorical_Feature.unique(),1):
            q1, q3= np.percentile(  df[(df['Categorical_Feature'] == i)]['price']  ,[25,75])
            iqr = q3 - q1
            lower_bound = q1 - (1.50 * iqr)
            upper_bound = q3 + (1.50 * iqr)
            a = df.shape[0]
            df = df.drop( 
                        (df[(df['price'] > upper_bound)].index | df[(df['price'] < lower_bound)].index ) & 
                        df[(df['Categorical_Feature'] == i)].index 
                    )
            tmp_df = df[(df['Categorical_Feature'] == i)]['price']
            if print_samples:
                print('%3d_ %-30s'%(cou,str(i)), 'removed:%-8d'%(a - df.shape[0]),'remaining: %-6d'%tmp_df.shape[0],
                     '  mean:%-.3f'%tmp_df.mean() ,' price_interval:[%-.3f,%-.3f]'%(tmp_df.min() , tmp_df.max())
                     )

    # Applying Label-Encoding for "Categorical_Feature"
    from sklearn.preprocessing import LabelEncoder
    df['Categorical_Feature_LE'] = LabelEncoder().fit_transform(df['Categorical_Feature']).astype(int)

    # remove 0's prices which causes MAPE inf errors
    df = df[ ~(df.price == 0) ]

    ######################################################################
    ###### scalling: applying log and [0, 1] scaling
    ########## #####################################
    df['price_scaled'] =  log_scaling_transformation(df["price"])


    ######################################################################
    ###### calculate mean of each "Categorical_Feature"
    ########## ########################################
    df['avg_true'] = get_mean_values(df, 'price_scaled')

    ######################################################################
    ###### Applying Mean_shiftting transformation
    ########## ##################################
    df, avg_shifting_dic, col_name_lst = apply_shiftting(df[['Categorical_Feature_LE', 'price_scaled', 'avg_true']], data_set_name)

    ######################################################################
    ###### 0-1 scalling
    ########## ########
    df, zero_one_shifting_dic = scalling_shifted_prices(df, col_name_lst, scaling_lvl2)


    ####################################################################################
    ###### Applying Variance-reduction transformation for each shifted value
    ########## #############################################################
    df, col_name_lst = apply_transformation(df, col_name_lst)

    if 'price_0.0_1.0' in col_name_lst: # 'price_0.0_1.0' raises "hex.gram.Gram$NonSPDMatrixException" error with GLM model
        col_name_lst.remove('price_0.0_1.0')

    return df, avg_shifting_dic, zero_one_shifting_dic, col_name_lst

def save_regression_csv(lvl_2_df , setting_str, n, shift, zero_one_scaler, avg_shifting_dic, train_index, test_index, data_set_name, x_shape):
    '''
    parameters:
        lvl_1_prediction: model prediction on test-set
        lvl_2_df:         dataset used in lvl_2_regression
        data_set_name:    name of passed data-set
        setting_str:      string of dataset, lvl_1 regressor, feature engineering method and timestamp
        n:                value of transformation coefficient applied
        train:            training indices
        test:             testing indices
    
    saves the lvl_2_regression to csv file and make lvl_2 regression tasks
    
    '''
    # calculate the Categorical-mean of "price_shift_transform" column
    lvl_2_df['avg_shifted'] = get_mean_values( lvl_2_df, 'y_transform' )

    # calculate the inverse of the variance-reduced prices
    lvl_2_df['y_pred_inv_Eq'] = (n+1)*lvl_2_df['y_pred_transform'] - n*lvl_2_df['avg_shifted']

    # save dataframe to the disk
    df_path = 'csv/%s/%s.csv'%(data_set_name,setting_str)
    lvl_2_df.to_csv(df_path, sep=',')

    # Back-Transformation score
    if shift != 'Orig': # shift is not the orig
        B_transformation = get_score_string(lvl_2_df['y_shift'], lvl_2_df['y_pred_inv_Eq'], x_shape = x_shape)

        # Apply B-shifting, then get the score
        # print('df.Columns ----------' , lvl_2_df.columns )

        a, b, str_ = Back_shift(lvl_2_df.rename(columns = {'y_pred_inv_Eq':'y_pred'}, inplace=False),
                                                                                   zero_one_scaler,
                                                                                   avg_shifting_dic,
                                                                                   ' B_transformation ' + setting_str )

        B_transformation_Bshift_1 = get_score_string(y_true = lvl_2_df['y_true'],
                                                   y_pred = a,
                                                   x_shape = x_shape)

        B_transformation_Bshift_2 = get_score_string(y_true = lvl_2_df['y_true'],
                                                   y_pred = b,
                                                   x_shape = x_shape)

        write_to_file(data_set_name,'\n    %-20s'%'B-Transform score', B_transformation)
        write_to_file(data_set_name,str_.replace('$',''))
        write_to_file(data_set_name,'    %-20s'%'B-Trans_shift MAE', B_transformation_Bshift_1)
        write_to_file(data_set_name,'    %-20s'%'B-Trans_shift G-MAE', B_transformation_Bshift_2,'\n')

    else:
        # no shift is applied, then calculate the score
        B_transformation = get_score_string(lvl_2_df['y_true'], lvl_2_df['y_pred_inv_Eq'], x_shape = x_shape)
        write_to_file(data_set_name,'\n    B-Transform score   ', B_transformation, '\n')

    from ._lvl2_regression import lvl2_regression
    lvl2_regression( df_path,
                     n, shift,
                     zero_one_scaler, avg_shifting_dic,
                     train_index, test_index,
                     data_set_name )


















