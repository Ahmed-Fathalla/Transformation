# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: Scalling and transformation methods applied in this work
"""

import numpy as np
import pandas as pd

transformation_coef_lst = [0.0,1.0]
price_transformation_coef_lst = ['price_scaled_'+str(i) for i in transformation_coef_lst]

def get_mean_values(df, price_column_name):
    '''
    get the mean value of each category of the 'Categorical_Feature'
    'price_scaled' is the original price, where no transformation is applied on

    parameter
        dataframe of 'Categorical_Feature' and 'price_scaled'

    return
        Apply log_ and Scalling_ transformation on the price_dataseries
    '''
    group = df.groupby(['Categorical_Feature_LE']).agg({ price_column_name: ['mean'] })
    group.columns = [ 'avg' ]
    group.reset_index(inplace=True)
    df = pd.merge(df, group, on=['Categorical_Feature_LE'], how='left')
    return list(df['avg'])

def Transform(df, n):
    '''
    n is the transformation coefficient
    '''
    return (df['price'] + n*df['avg'])/(n+1)

def log_scaling_transformation(price_dataseries):
    '''
    parameter
        price_dataseries
    return
        Apply log_ and Scalling_ transformation on the price_dataseries
    '''
    from sklearn.preprocessing import MinMaxScaler
    print('\n\n%-26s[%-.5f, %-.5f]'%('Original prices Range',price_dataseries.min(), price_dataseries.max()))
    # Log Transformation
    price_log = np.log(price_dataseries+1)

    print('%-26s[%-.5f, %-.5f]'%('After Log Transformation',price_log.min(), price_log.max()))

    # MinMax Scaling
    min_max_var = 0.01
    target_scaler = MinMaxScaler(feature_range=(min_max_var, 1))

    print('%-26s[%-.5f, %-.5f]'%('After 0-1 Transformation',min_max_var, 1))
    return target_scaler.fit_transform(np.array(price_log).reshape(-1,1))

def apply_transformation(df, col_lst):
    col_name_lst = []
    for col_name in col_lst:
        tmp_df = df[['Categorical_Feature_LE', col_name]]
        tmp_df['avg'] = get_mean_values( tmp_df, col_name )
        for n in transformation_coef_lst:
            col_name_lst.append( col_name+'_'+str(n) )
            df[col_name_lst[-1]] = Transform(tmp_df[[col_name,'avg']].rename(columns = {col_name:'price'}),
                                             n)

    return df, col_name_lst

def scalling_shifted_prices(df, col_name_lst, scaling_lvl2):
    from sklearn.preprocessing import MinMaxScaler
    if scaling_lvl2:
        shift_dic = {}
        for col in col_name_lst:
            if 'Orig' not in col:
                df[col+'_orig'] = df[col]

                shift_dic[col] = MinMaxScaler(feature_range=(0.001, 1))
                # shift_dic[col].fit(df[col])
                df[col] = shift_dic[col].fit_transform(df[col].values.reshape(-1,1))
            else:
                shift_dic[col] = None
    else:
        for col in col_name_lst:
            if 'Orig' not in col:
                df[col+'_orig'] = df[col]
        shift_dic = None

    return df, shift_dic
