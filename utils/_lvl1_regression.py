# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: Sklearn and H2O regression methods for the first level regression
"""

import time
import pandas as pd
import h2o
from scipy.sparse import hstack
import pickle # for saving ML models to the disk
import traceback

from ._utils import *
from ._time_utils import get_timestamp
from ._regressors import H2O_model_lst, SK_model_lst, SK_lvl1_model_dic
from ._scoring_metrics import *
from ._shifting_avgerage import Back_shift
from ._write_to_file import write_to_file
from ._transformation import get_mean_values
from ._config import get_lvl1_training_results
from ._feature_eng import get_Embedding_matrix
try:
    from ._statistical_models import GEE_, mixedlm_
except ImportError as exc:
    print('Cannot imoprt:\n',traceback.format_exc())


def get_scale_range(ds):
    return '[%-5s, %-5s]'%(str(ds.min())[:5],str(ds.max())[:5])

def check_shift_n(df, shift, n, zero_one_scaler, avg_shifting_dic, train_index, test_index, setting_str_, data_set_name):

    # 'y_true'      -> 'price_Orig_0.0', Original prices
    # 'y_shift'     -> 'price_X.X_orig', Shifted prices: price after applying mean-shift on the original price
    # 'y_transform' -> 'price_X.X_X.X', Transformed prices: after applying variance-reduction on the 0-1 scaled shifted prices

    tmp_df                = df[['Categorical_Feature_LE','avg_true']]
    tmp_df['y_true']      = df['price_Orig_0.0']

    if n: # n>0 for any value of shift

        tmp_df['y_shift']     = df['price_Orig_0.0' if shift=='Orig' else 'price_%s_orig'%shift]
        tmp_df['y_transform'] = df['price_%s_%s'%(shift,str(n))]  # price_shift_transform
        tmp_df['y_pred_transform'] = df['y_pred']
        save_regression_csv( tmp_df, setting_str_, n, shift,
                             zero_one_scaler, avg_shifting_dic[shift],
                             train_index, test_index, data_set_name, x_shape = None ) # x.shape

    elif shift != 'Orig': # n==0 & shift is not the orig

        tmp_df['y_shift']     = df['price_%s_orig'%shift]
        tmp_df['y_pred'] = df['y_pred']
        tmp_df['y_shift_scaled'] = df['price_%s_0.0'%shift]
        tmp_df['y_pred_Single_MAE'], tmp_df['y_pred_MAE_Group'], str_ = Back_shift( tmp_df,
                                                                                    zero_one_scaler,
                                                                                    avg_shifting_dic[shift],
                                                                                    ' lvl_1 ' + setting_str_)

        write_to_file(data_set_name, str_.replace('$',''), '%-33s %s'%('regressor output After  B-shift ',get_scale_range(df['y_pred'])))

        write_to_file(data_set_name,'    %-20s'%'Back_shift MAE',
                      get_score_string(y_true = tmp_df['y_true'].iloc[test_index],
                                       y_pred = tmp_df['y_pred_Single_MAE'].iloc[test_index] ) )

        write_to_file(data_set_name,'    %-20s'%'Back_shift G-MAE',
                      get_score_string(y_true = tmp_df['y_true'].iloc[test_index],
                                       y_pred = tmp_df['y_pred_MAE_Group'].iloc[test_index] ) )

        write_to_file(data_set_name,'')

    else: # n==0 & shift==0
        write_to_file(data_set_name,'')


#################################################
# SKlearn level_1 regression
############################
def SK_lvl1(x, df, price_lst, avg_shifting_dic, zero_one_scaler_dic, setting_str, data_set_name, save_ML_model = True):
    write_to_file(data_set_name,'\nStarting SKlearn_%s Exp at'%data_set_name,get_timestamp(1), ', Dataframe length:', df.shape[0], '\n')
    train_index, test_index= get_train_test_indexes(df['Categorical_Feature_LE'])
    X_train, _, X_test, _ = train_test_split_( X_df = x , y_df = None, train_index = train_index, test_index = test_index)
    for price_shift_transform in price_lst:
        shift, n = price_shift_transform.split('_')[1:] #[float(i) for i in price_shift_transform.split('_')[1:]]
        n = float(n)

        str_ = '%s %s %s price_scale_'%(data_set_name, price_shift_transform, setting_str) + \
               '%s'%get_scale_range(df[price_shift_transform]) +\
               ' len_%-8s'%(str(df.shape[0]))
        write_to_file(data_set_name,'\nSKlearn_', str_, get_timestamp(1), '\n')

        y = df[price_shift_transform]
        _, y_train, _, y_test = train_test_split_( X_df = None , y_df = y, train_index = train_index, test_index= test_index)

        mod_lst = [*SK_model_lst[:-1], SK_lvl1_model_dic[data_set_name]]
        for (model, model_name) in mod_lst:
            setting_str_ = get_Exp_str(str_, model_name, x.shape[0])

            start_time = time.time()
            model.fit(X_train, y_train)
            
            try:
                if save_ML_model:pickle.dump(model, open('models/%s_%s.sav'%(data_set_name, model_name), 'wb'))
            except FileNotFoundError:
                os.mkdir('models')
                pickle.dump(model, open('models/%s_%s.sav'%(data_set_name, model_name), 'wb'))
                
            fitting_time = str( (time.time() - start_time)/60)[:5]
            df['y_pred'] = model.predict(x)#.reshape(-1, 1)

            # get training score
            # ==================
            output_str = ''
            if get_lvl1_training_results:
                tmp_df = df[['Categorical_Feature_LE',price_shift_transform, 'y_pred']].iloc[train_index].\
                    rename(columns = {price_shift_transform:'y_true_shifted'}, inplace = False)# if n else df[['price_0_0']].iloc[train_index]
                output_str = get_score_________(tmp_df,'    Train %-15s'%model_name,
                         data_set_name, get_str = 1, x_shape = x.shape, timestamp=True)+'\n'

            # get test score
            # ==============
            tmp_df = df[['Categorical_Feature_LE',price_shift_transform, 'y_pred']].iloc[test_index].\
                rename(columns = {price_shift_transform:'y_true_shifted'}, inplace = False)# if n else df[['price_0_0']].iloc[test_index]
            get_score_________(tmp_df,'%s    %s %-15s'%(output_str, fitting_time, model_name),
                      data_set_name, x_shape = x.shape, timestamp=False)

            ls = list(set(['Categorical_Feature_LE','price_Orig_0.0', 'avg_true',
                              'price_Orig_0.0' if shift=='Orig' else 'price_%s_orig'%shift,
                              'price_%s_0.0'%shift, 'price_%s_%s'%(shift,str(n)), 'y_pred']))
            check_shift_n(df[ls],
                          shift, n,
                          zero_one_scaler_dic['price_%s'%(shift)] if zero_one_scaler_dic != None else None, avg_shifting_dic,
                          train_index, test_index, setting_str_, data_set_name)

        write_to_file(data_set_name,'\n##################################################################\n')

#################################################
# H2O level_1 regression
########################
def H2O_lvl1(x, df, price_lst, avg_shifting_dic, zero_one_scaler_dic, setting_str, data_set_name, save_ML_model = True):
    write_to_file(data_set_name,'\nStarting H2O_%s Exp at'%data_set_name,get_timestamp(1), ', Dataframe length:', df.shape[0], '\n')
    try:
        if str(h2o.connection()) == '<H2OConnection closed>' or str(h2o.connection()) == 'None':
            write_to_file(data_set_name,'********** creating new instance')
            h2o.init(min_mem_size = "230G") # h2o.init(ip="127.0.0.1",port="54321",min_mem_size = "1G")
            h2o.remove_all
    except Exception as exc:
        write_to_file(data_set_name,'********** cannot create new instance\n', str(exc) )

    # h2o.cluster().show_status()
    h2o.no_progress()

    train_index, test_index= get_train_test_indexes(df['Categorical_Feature_LE'])
    feature_lst = ['feat_'+str(o) for o in range(x.shape[1])]
    new_df = pd.DataFrame( data=x, columns=feature_lst)
    new_df = pd.concat([new_df, df[price_lst]],axis=1)

    x_train = h2o.H2OFrame(new_df.loc[train_index])
    x_test =  h2o.H2OFrame(new_df.loc[test_index])

    for price_shift_transform in price_lst:
        shift, n = price_shift_transform.split('_')[1:] #[float(i) for i in price_shift_transform.split('_')[1:]]
        n = float(n)

        str_ = '%s %s %s price_scale_'%(data_set_name, price_shift_transform, setting_str) + \
               '%s'%get_scale_range(df[price_shift_transform]) +\
               ' len_%-8s'%(str(df.shape[0]))
        write_to_file(data_set_name,'\nH2O_', str_, get_timestamp(1), '\n')
        y = price_shift_transform
        for i in range(len(H2O_model_lst)):

            start_time = time.time()
            H2O_model_lst[i].train(x=feature_lst, y=y, training_frame=x_train)
            fitting_time = str( (time.time() - start_time)/60)[:5]
            
            model = H2O_model_lst[i].model_id
            model_name = model.split('_')[0]
            
            if save_ML_model:w = h2o.save_model(H2O_model_lst[i], path = 'models/%s'%data_set_name)

            df['y_pred'] = h2o.get_model(model).predict(h2o.H2OFrame(new_df)).as_data_frame()['predict'].tolist()

            # get training score
            # ==================
            output_str = ''
            mape_res = MAPE( x_train[y].as_data_frame()[y].tolist(), df['y_pred'].loc[train_index] )
            if get_lvl1_training_results:
                model_dic = ( (' Train %-15s'%model_name,
                               get_h2o_score_string(h2o.get_model(model).model_performance(x_train), mape_res, n, [new_df.shape[0], len(feature_lst)], timestamp=True ),
                               h2o.get_model(model).model_performance(x_train).mae()
                              )  )
                output_str = get_h2o_score(data_set_name, model_dic[0],model_dic[1], get_str=1)+'\n'

            # get test score
            # ==============
            mape_res = MAPE( x_test[y].as_data_frame()[y].tolist(), df['y_pred'].loc[test_index] )
            setting_str_ = get_Exp_str(str_, model_name, x.shape[0])
            model_dic = ( ('%-14s'%model_name,
                           get_h2o_score_string(h2o.get_model(model).model_performance(x_test), mape_res, n, [new_df.shape[0], len(feature_lst)] ),
                           h2o.get_model(model).model_performance(x_test).mae()
                          )  )

            get_h2o_score(data_set_name, "%s    %s %s"%(output_str, fitting_time,model_dic[0]),model_dic[1])

            ls = list(set(['Categorical_Feature_LE','price_Orig_0.0', 'avg_true',
                  'price_Orig_0.0' if shift=='Orig' else 'price_%s_orig'%shift,
                  'price_%s_0.0'%shift, 'price_%s_%s'%(shift,str(n)), 'y_pred']))
            check_shift_n(df[ls],
                          shift, n,
                          zero_one_scaler_dic['price_%s'%(shift)], avg_shifting_dic,
                          train_index, test_index, setting_str_, data_set_name)

        write_to_file(data_set_name,'\n##################################################################\n')

#################################################
# Statistical models regression
########################

# Getting non-colinear features for statistical models
def get_non_colinear_DataFrame(x, y_col, df):
    try:
        if str(h2o.connection()) == '<H2OConnection closed>' or str(h2o.connection()) == 'None':
            print('********** creating new instance')
            h2o.init(min_mem_size = "230G") # h2o.init(ip="127.0.0.1",port="54321",min_mem_size = "1G")
            h2o.remove_all
            h2o.no_progress()
    except Exception as exc:
        print('********** cannot create new instance\n', str(exc) )
    feature_lst = ['feat_'+str(o) for o in range(x.shape[1])]

    new_df = pd.DataFrame(x , columns=feature_lst)
    new_df['y'] = df[y_col]
    new_df['Categorical_Feature_LE'] = df['Categorical_Feature_LE']

    from h2o.estimators.glm import H2OGeneralizedLinearEstimator
    Glm_model = H2OGeneralizedLinearEstimator(family= "gaussian", compute_p_values = True,remove_collinear_columns = True, lambda_ = 0,seed=1)
    start_time = time.time()
    Glm_model.train(x=feature_lst, y='y', training_frame=h2o.H2OFrame(new_df) )
    fitting_time = str( (time.time() - start_time)/60)[:5]
    model = Glm_model.model_id
    non_col_feature_lst  = [feature_name for feature_name, feature_coef in Glm_model.coef().items() if feature_coef !=0 ]
    new_df = new_df[['y', 'Categorical_Feature_LE', *non_col_feature_lst[1:] ]]
    return new_df

def Statistical_models(x, y , df, data_set_name, save_ML_model):
    write_to_file(data_set_name,'\nStarting Statistical_models_%s Exp at'%data_set_name,get_timestamp(1), ', Dataframe length:', df.shape[0], '\n')
    new_df  =  get_non_colinear_DataFrame(x, y, df)
    GEE_(new_df, data_set_name, save_ML_model = save_ML_model)
    mixedlm_(new_df, data_set_name, save_ML_model = save_ML_model)

###############################
# LSTM Model
############
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, GRU, Embedding, Dropout, Input
from keras.layers.merge import concatenate
from keras.models import Model

def fit_LSTM(X_train, y_train, X_test, y_test, one_hot_features, vocab_size, maxlen, batch_size, epochs, model_name, data_set_name):

    # one hot encoding category_name model
    # ====================================
    category_model_input = Input(shape=(one_hot_features,), name="category_Input")
    category_model_out = Dense(128, activation='relu')(category_model_input)

    # txt_model
    # =========
    txt_model_input = Input( shape=(maxlen,) , name='txt_Input' )
    txt_model_in = Embedding(
                            input_dim    = vocab_size,
                            output_dim   = 128,
                            input_length = maxlen,
                            mask_zero = True
                          )(txt_model_input)
    txt_model_in = GRU(128, return_sequences=False)(txt_model_in)
    txt_model_out = Dense(64, activation='relu')(txt_model_in)

    ###########
    # out_model
    # =========
    out_model = concatenate([txt_model_out, category_model_out], axis=1)
    out_model = Dense(64, activation='relu')(out_model)
    out_model = Dense(1, activation='sigmoid', name='Output_layer')(out_model)
    model = Model( inputs=[txt_model_input, category_model_input], outputs = out_model )
    model.compile(loss = MAE_keras, optimizer = 'Adam', metrics=[R2_keras, MAE_keras, RMSE_keras, MAPE_keras])
    checkpoint = ModelCheckpoint(filepath='hdf5/%s_ep_{epoch:2d}.hdf5'%model_name,
                                  monitor='val_MAE_keras',verbose=0,save_best_only=True,mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_MAE_keras', factor=0.3,
                              patience=2, min_lr=0.000001, verbose=0)
    es = EarlyStopping(monitor='val_MAE_keras', mode='min', verbose=0)
    callbacks_list = [checkpoint, reduce_lr, es]
    start_time = time.time()
    history = model.fit(
                                x = [X_train[:,: -one_hot_features],X_train[:,-one_hot_features:]] ,
                                y = y_train ,
                                batch_size = batch_size,
                                epochs = epochs,
                                verbose = 1,
                                callbacks=callbacks_list,
                                validation_data=(  [X_test[:,: -one_hot_features ],X_test[:,-one_hot_features:]] , y_test)
                               )
    fitting_time = str( (time.time() - start_time)/60)[:5]
    write_to_file(data_set_name, '\n    >>>> fitting time: %s     '%fitting_time, get_timestamp(1))
    for i in range(len(history.history['MAE_keras'])):
        s = 'Train R2:'+ str(history.history['R2_keras'][i]*100)[:6]+\
        '  MAE:' + str(history.history['MAE_keras'][i])[:6] +\
        '  RMSE:' + str(history.history['RMSE_keras'][i])[:6] +\
        '  MAPE:' + str(history.history['MAPE_keras'][i])[:6]+ '\n\t      ' +\
        'Test  R2:'+ str(history.history['val_R2_keras'][i]*100)[:6]+\
        '  MAE:' + str(history.history['val_MAE_keras'][i])[:6] +\
        '  RMSE:' + str(history.history['val_RMSE_keras'][i])[:6] +\
        '  MAPE:' + str(history.history['val_MAPE_keras'][i])[:6]
        if history.history['val_MAE_keras'][i] == min(history.history['val_MAE_keras']):
            write_to_file(data_set_name, '\tEP_%-3d%s'%(i+1,s), '  ***********')
        else:
            write_to_file(data_set_name, '\tEP_%-3d%s'%(i+1,s))
    write_to_file(data_set_name, '')
    epoch_ = len(history.history['MAE_keras'])
    return model, epoch_

def LSTM_lvl1(one_hot_matrix, df, price_lst, avg_shifting_dic, zero_one_scaler_dic, setting_str, data_set_name, maxlen = 300, batch_size = 2048, epochs = 10):

    data_set_name = 'LSTM_' + data_set_name

    train_index, test_index = get_train_test_indexes(df['Categorical_Feature_LE'])
    
    Embedding_matrix, vocab_size = get_Embedding_matrix(df['description'], maxlen=maxlen)
    x = hstack(( Embedding_matrix, one_hot_matrix )).toarray()
    number_of_categories = one_hot_matrix.shape[1]
    
    X_train, _, X_test, _ = train_test_split_( X_df = x , y_df = None, train_index = train_index, test_index= test_index)

    for price_shift_transform in price_lst:
        shift, n = price_shift_transform.split('_')[1:] #[float(i) for i in price_shift_transform.split('_')[1:]]
        n = float(n)

        str_ = '%s %s price_scale: '%(data_set_name, price_shift_transform) + get_scale_range(df[price_shift_transform]) +\
               ' %s len_%-8s'%(setting_str.split()[1], str(df.shape[0]))
        write_to_file(data_set_name,'\n', str_, get_timestamp(1), '\n')

        y = df[price_shift_transform]
        _, y_train, _, y_test = train_test_split_( X_df = None , y_df = y, train_index = train_index, test_index= test_index)

        model_name = 'LSTM_' + data_set_name + ' ' + price_shift_transform + ' ' + get_timestamp()

        model, ep = fit_LSTM(X_train, y_train, X_test, y_test, number_of_categories, vocab_size, maxlen, batch_size, epochs, model_name, data_set_name)


        start_time = time.time()
        model_name = 'hdf5/%s_ep_%2s.hdf5'%( model_name, str(ep))

        output_str = ''
        df['y_pred'] = get_LSTM_prediction( x, model, model_name, number_of_categories)

        fitting_time = str( (time.time() - start_time)/60)[:5]


        # get training score
        # ==================
        output_str = ''
        if get_lvl1_training_results:
            tmp_df = df[['Categorical_Feature_LE',price_shift_transform, 'y_pred']].iloc[train_index].\
                rename(columns = {price_shift_transform:'y_true_shifted'}, inplace = False)# if n else df[['price_0_0']].iloc[train_index]
            output_str = get_score_________(tmp_df,'        Train ',
                     data_set_name, get_str = 1, x_shape = x.shape, timestamp=True)+'\n'

        # get test score
        # ==============
        tmp_df = df[['Categorical_Feature_LE',price_shift_transform, 'y_pred']].iloc[test_index].\
            rename(columns = {price_shift_transform:'y_true_shifted'}, inplace = False)# if n else df[['price_0_0']].iloc[test_index]
        get_score_________(tmp_df,'%s        %s '%(output_str, fitting_time),
                  data_set_name, x_shape = x.shape, timestamp=False)

        ls = list(set(['Categorical_Feature_LE','price_Orig_0.0', 'avg_true',
                          'price_Orig_0.0' if shift=='Orig' else 'price_%s_orig'%shift,
                          'price_%s_0.0'%shift, 'price_%s_%s'%(shift,str(n)), 'y_pred']))
        check_shift_n(df[ls],
                      shift, n,
                      zero_one_scaler_dic['price_%s'%(shift)], avg_shifting_dic,
                      train_index, test_index, setting_str, data_set_name)


        #####################################################
        # clearning the memory
        from keras import backend as K
        import keras.backend.tensorflow_backend
        K.clear_session()
        if keras.backend.tensorflow_backend._SESSION:
            import tensorflow as tf
            tf.reset_default_graph()
            keras.backend.tensorflow_backend._SESSION.close()
            keras.backend.tensorflow_backend._SESSION = None
        #####################################################

        write_to_file(data_set_name,'\n##################################################################\n')