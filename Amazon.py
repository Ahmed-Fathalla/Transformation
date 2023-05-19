#
from utils import *
data_set_name = 'Amazon'# os.path.basename(sys.argv[0]).split('.')[0]

try:
    # Reading the dataframe
    df = pd.read_csv('%s/%s.csv'%(utl.dataset_path, data_set_name),sep=',')

    # Set the Categorical feature that transformation is applied on
    df.rename( columns= {'ds_name':'Categorical_Feature'}, inplace=True )

    for outlier_type in ['None','IQR' ][:1]:

        curr_df, avg_shifting_dic, zero_one_scaler_dic, price_lst = utl.handle_df(df[['Categorical_Feature', 'price']], outlier_type,
                                                                                data_set_name, scaling_lvl2 = True)

        curr_df = pd.merge( df[['price', 'description', 'Categorical_Feature']], curr_df,
                            left_on=df.index, right_on=curr_df.index, how='right')

        one_hot_matrix = utl.get_One_Hot_Encoding(curr_df['Categorical_Feature_LE'])

        # for feature_eng in [utl.TF_IDF, utl.BOW ][:1]:
        #     x = hstack(( feature_eng(curr_df['description']), one_hot_matrix )).toarray()
        #
        #     curr_df.drop(['description'], axis=1, inplace=True)
        #     setting_str = ' %-6s outlier_%-4s '%(feature_eng.__name__, outlier_type)
        #
        #     ###########################################
        #     ###### SKlearn Regresors
        #     ########################
        #     # print( 'price_lst = ' , price_lst )
        #     if utl.Exp_options['run_SK_models']:utl.SK_lvl1(x, curr_df.copy(), price_lst,
        #                                         avg_shifting_dic, zero_one_scaler_dic,
        #                                         setting_str, data_set_name, save_ML_model = utl.save_ML_model)
        #
        #     #######################################
        #     ###### H2O Regresors
        #     ####################
        #     if utl.Exp_options['run_H2O_models']:utl.H2O_lvl1( x, curr_df.copy(), price_lst,
        #                                                        avg_shifting_dic, zero_one_scaler_dic,
        #                                                        setting_str, data_set_name, save_ML_model = utl.save_ML_model
        #                                                       )
        #
        #     #######################################
        #     ###### Statistical moldes
        #     #########################
        #     # if utl.Exp_options['run_Statistical_models']:
        #     #     utl.Statistical_models(x, 'price_Orig_0.0', curr_df[['price_Orig_0.0', 'Categorical_Feature_LE']].copy(),
        #                            data_set_name = data_set_name, save_ML_model = utl.save_ML_model)

        #######################################
        ###### LSTM_Regression
        #############################

        data_set_name = 'LSTM_' + data_set_name
        if utl.Exp_options['run_LSTM_model']:
            utl.LSTM_lvl1(one_hot_matrix = one_hot_matrix,
                      df = curr_df.copy(),
                      price_lst = price_lst,
                      avg_shifting_dic = avg_shifting_dic,
                      zero_one_scaler_dic = zero_one_scaler_dic,
                      setting_str = 'LSTM outlier_%-4s'%outlier_type,
                      data_set_name = data_set_name,
                      epochs = 1)

except Exception as exc:
    utl.write_to_file(data_set_name, '\n**** Err: ' + utl.get_timestamp() +  ' \n',traceback.format_exc())


