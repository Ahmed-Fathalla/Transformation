# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: Sklearn and H2O regression methods for the second level regression
"""

import h2o
import pandas as pd
import operator
import time
from ._utils import train_test_split_
from ._time_utils import get_timestamp
from ._scoring_metrics import *
from ._shifting_avgerage import Back_shift
from ._feature_eng import lvl2_feature_engineering
from ._regressors import H2O_model_lst, SK_model_lst
from ._write_to_file import write_to_file
from ._config import get_lvl2_training_results

def get_sorted_tuble_lst(data_set_name, tub_lst):
    '''
    parameter:
        tub_lst is a list of tubles, each of ( Model_Name, Model_performance_results, Model_MAE_result )

    return:
        sorts tub_lst according to Model_MAE_result in descending order, and call "get_h2o_score" for each tuble.
    '''
    for i in sorted(tub_lst, key=operator.itemgetter(2), reverse=False):
        get_h2o_score(data_set_name, i[0], i[1])

def lvl2_regression(df_path, n, shift, zero_one_scaler, avg_shifting_dic, train_index, test_index, data_set_name ):
    '''
    performs lvl2_regression for both of sklearn and h2o models

    '''
    start_time = time.time()
    sk_model_results_dict  = sk_regression(df_path,
                                           n, shift,
                                           zero_one_scaler, avg_shifting_dic,
                                           train_index, test_index,
                                           data_set_name)

    h2o_model_results_dict = h2o_regression(df_path,
                                            n, shift,
                                            zero_one_scaler, avg_shifting_dic,
                                            train_index, test_index,
                                            data_set_name)
    # h2o_model_results_dict = {'y_pred_transform':[],
    #                       'y_pred_inv_Eq':[] }

    fitting_time = str( (time.time() - start_time)/60)[:5]
    write_to_file(data_set_name, '\n    >>>> lvl_2 fitting time:', fitting_time)

    method_1 = [*sk_model_results_dict['y_pred_transform'], *h2o_model_results_dict['y_pred_transform']]
    write_to_file(data_set_name, '   '*2,df_path ,' lvl2_regression_len:',sk_model_results_dict['df_shape'], get_timestamp(2),'\n\n\t         *** Method_1: y_pred_transform -> y_true:\n')
    get_sorted_tuble_lst(data_set_name, method_1)

    method_2 = [*sk_model_results_dict['y_pred_inv_Eq'], *h2o_model_results_dict['y_pred_inv_Eq']]
    write_to_file(data_set_name, '\n\t         *** Method_2: y_inverse_eq -> y_true:\n')
    get_sorted_tuble_lst(data_set_name, method_2)


    write_to_file(data_set_name, '\n' ,'-'*50, '\n')

def sk_regression(df_path, n, shift, zero_one_scaler, avg_shifting_dic, train_index, test_index, data_set_name ):

    # df has the following colunms
    # ============================
    # 'Categorical_Feature_LE', 'avg_shifted'
    # 'y_true'           -> 'price_0.0_0.0', Original prices
    # 'y_shift'          -> 'price_X.X_0.0', Shifted prices: price after applying mean-shift on the original price
    # 'y_transform'      -> 'price_X.X_X.X', Transformed prices: after applying variance-reduction on the shifted prices
    # 'y_pred_transform' -> model prediction on "y_transform"
    # 'y_pred_inv_Eq'    -> Back-variance-Transformation of "y_pred_transform"

    df = pd.read_csv(df_path, delimiter=',', usecols = ['Categorical_Feature_LE',
                                                        'avg_shifted','avg_true',
                                                        'y_true', 'y_shift', 'y_transform',
                                                        'y_pred_transform','y_pred_inv_Eq'
                                                        ])

    y = df['y_shift']

    method_lst = ['y_pred_transform', 'y_pred_inv_Eq']
    model_results_dict = {}

    for method in method_lst:
        model_results_dict[method] = []
        x, _ = lvl2_feature_engineering(df[['avg_shifted', method]].copy(),col = method)
        X_train, y_train, X_test, y_test = train_test_split_( x , y, train_index, test_index )

        for (model, model_name) in SK_model_lst:

            start_time = time.time()
            model.fit(X_train, y_train)
            fitting_time = str( (time.time() - start_time)/60)[:5]

            df['y_pred'] = model.predict(x)

            # Calculate the back-shift of the predicted data
            if shift != 'Orig':
                df['y_pred_Single_MAE'], df['y_pred_MAE_Group'], str_ = \
                            Back_shift( df[['Categorical_Feature_LE', 'y_pred', 'y_true', 'avg_true', 'y_shift']],
                                        zero_one_scaler,
                                        avg_shifting_dic,
                                        model_name + '^' + df_path[:-4]) # "^" is used later as a splitting character in _shifting_avgerage.Back_shift
                output_str = str_.replace('$',' '*4) + '\n'
            else:
                output_str = ''

            if get_lvl2_training_results:
                output_str += '\t\tTrain %-14s '%model_name+get_score_string( y_train, df['y_pred'].iloc[train_index] ) + '^'#[:-19].strip() + '^' # splitting character
                if shift != 'Orig': # shift is not the orig
                    output_str += '\t\t%-21s'%'train B-shift MAE'+\
                                 get_score_string( df.iloc[train_index]['y_true'],df.iloc[train_index]['y_pred_Single_MAE'])+ '^'+'\t\t%-21s'%'train B-shift G-MAE'+ \
                                 get_score_string( df.iloc[train_index]['y_true'],df.iloc[train_index]['y_pred_MAE_Group'])+ '^'

            output_str += '\t\t%s %-14s '%(fitting_time,model_name)+ get_score_string(y_test, df['y_pred'].iloc[test_index] )[:-19].strip()+' '*13+ get_timestamp(1)+ '^'
            if shift != 'Orig': # shift is not the orig
                output_str += '\t\t%-21s'%'test  B-shift MAE'+\
                             get_score_string( df.iloc[test_index]['y_true'],df.iloc[test_index]['y_pred_Single_MAE'])+ '^'+'\t\t%-21s'%'test  B-shift G-MAE'+ \
                             get_score_string( df.iloc[test_index]['y_true'],df.iloc[test_index]['y_pred_MAE_Group'])+ '^'

                model_results_dict[method].append((output_str,
                                                '',
                                                min(
                                                MAE_sklearn(df['y_true'].iloc[test_index],
                                                            df['y_pred_Single_MAE'].iloc[test_index]
                                                            ), # sort results by MAE
                                                MAE_sklearn(df['y_true'].iloc[test_index],
                                                        df['y_pred_MAE_Group'].iloc[test_index]
                                                        )
                                                )
                                              ))
            else:
                 model_results_dict[method].append((output_str,
                                                '',
                                                MAE_sklearn(df['y_true'].iloc[test_index],
                                                            df['y_pred'].iloc[test_index]
                                                            ), # sort results by MAE
                                              ))


    model_results_dict['df_shape'] = df.shape[0]
    return model_results_dict

def h2o_regression(df_path, n, shift, zero_one_scaler, avg_shifting_dic, train_index, test_index, data_set_name ):

    # df has the following colunms
    # ============================
    # 'Categorical_Feature_LE', 'avg_shifted'
    # 'y_true'           -> 'price_0.0_0.0', Original prices
    # 'y_shift'          -> 'price_X.X_0.0', Shifted prices: price after applying mean-shift on the original price
    # 'y_transform'      -> 'price_X.X_X.X', Transformed prices: after applying variance-reduction on the shifted prices
    # 'y_pred_transform' -> model prediction on "y_transform"
    # 'y_pred_inv_Eq'    -> Back-variance-Transformation of "y_pred_transform"

    df = pd.read_csv(df_path, delimiter=',', usecols = ['Categorical_Feature_LE',
                                                        'avg_shifted','avg_true',
                                                        'y_true', 'y_shift', 'y_transform',
                                                        'y_pred_transform','y_pred_inv_Eq'
                                                        ])
    y = 'y_shift'

    if str(h2o.connection()) == '<H2OConnection closed>' or str(h2o.connection())=='None':
        h2o.init(ip="127.0.0.1",port="54321",min_mem_size = "250G")
        h2o.remove_all
    h2o.no_progress()

    method_lst = ['y_pred_transform', 'y_pred_inv_Eq']
    model_results_dict = {}

    for method in method_lst:
        model_results_dict[method] = []
        base_models_ = []
        h2o_df, feature_lst = lvl2_feature_engineering(df[[y, 'avg_shifted', method]].copy(),col = method)
        train = h2o.H2OFrame(h2o_df.iloc[train_index])
        test =  h2o.H2OFrame(h2o_df.iloc[test_index])
        for i in range(len(H2O_model_lst)):
            start_time = time.time()
            # print( '================ ' , H2O_model_lst[i].model_id.split('_')[0] )
            H2O_model_lst[i].train(x=feature_lst, y=y, training_frame=train)
            fitting_time = str( (time.time() - start_time)/60)[:5]
            base_models_.append(H2O_model_lst[i].model_id)

        for model in base_models_:

            df['y_pred'] = h2o.get_model(model).predict(h2o.H2OFrame(h2o_df)).as_data_frame()['predict'].tolist()

            if shift != 'Orig':
                df['y_pred_Single_MAE'], df['y_pred_MAE_Group'], str_ = \
                            Back_shift( df[['Categorical_Feature_LE', 'y_pred', 'y_true', 'avg_true', 'y_shift']],
                                        zero_one_scaler,
                                        avg_shifting_dic,
                                        model.split('_')[0] + '^' + df_path[:-4]) # "^" is used later as a splitting character in _shifting_avgerage.Back_shift
                output_str = str_.replace('$',' '*4) + '\n'
            else:
                output_str = ''

            if get_lvl2_training_results:
                output_str += '\t\tTrain %-14s '%model.split('_')[0] + get_score_string( df.iloc[train_index][y], df['y_pred'].iloc[train_index] ) + '^' #[:-19].strip() + '^' # splitting character
                if shift != 'Orig': # shift is not the orig
                    output_str += '\t\t%-21s'%'train B-shift MAE'+\
                                 get_score_string( df.iloc[train_index]['y_true'],df.iloc[train_index]['y_pred_Single_MAE'])+ '^'+'\t\t%-21s'%'train B-shift G-MAE'+ \
                                 get_score_string( df.iloc[train_index]['y_true'],df.iloc[train_index]['y_pred_MAE_Group'])+ '^'

            output_str += '\t\t%s %-14s '%(fitting_time,model.split('_')[0])+ get_score_string(df.iloc[test_index][y], df['y_pred'].iloc[test_index] )[:-19].strip()+' '*13+ get_timestamp(1)+ '^'
            if shift != 'Orig': # shift is not the orig
                output_str += '\t\t%-21s'%'test  B-shift MAE'+\
                             get_score_string( df.iloc[test_index]['y_true'],df.iloc[test_index]['y_pred_Single_MAE'])+ '^'+'\t\t%-21s'%'test  B-shift G-MAE'+ \
                             get_score_string( df.iloc[test_index]['y_true'],df.iloc[test_index]['y_pred_MAE_Group'])+ '^'

                model_results_dict[method].append((output_str,
                                                '',
                                                min(
                                                MAE_sklearn(df['y_true'].iloc[test_index],
                                                            df['y_pred_Single_MAE'].iloc[test_index]
                                                            ), # sort results by MAE
                                                MAE_sklearn(df['y_true'].iloc[test_index],
                                                        df['y_pred_MAE_Group'].iloc[test_index]
                                                        )
                                                )
                                              ))
            else:
                 model_results_dict[method].append((output_str,
                                                '',
                                                MAE_sklearn(df['y_true'].iloc[test_index],
                                                            df['y_pred'].iloc[test_index]
                                                            ), # sort results by MAE
                                              ))
    return model_results_dict

