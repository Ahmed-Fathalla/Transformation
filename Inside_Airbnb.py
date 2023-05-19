#
from utils import *
data_set_name = 'Inside_Airbnb'# os.path.basename(sys.argv[0]).split('.')[0]

try:
    # Reading the dataframe
    df = pd.read_csv('transformation_ds/%s.csv'%data_set_name,encoding="ISO-8859-1")#[:1000]
    lst = ['room_type', 'property_type', 'Categorical_Feature'] # 'street', 'city',

    for outlier_type in ['None','IQR' ][:1]:
        curr_df, avg_shifting_dic, zero_one_scaler_dic, price_lst = handle_df(df[['Categorical_Feature', 'price']], outlier_type, data_set_name)

        curr_df = pd.merge( df[[*lst, 'price', 'description']],
                            curr_df,
                            left_on=df.index, right_on=curr_df.index, how='right')

        one_hot_matrix = get_One_Hot_Encoding(curr_df[lst])

        for feature_eng in [TF_IDF, BOW ][:1]:
            x = hstack(( feature_eng(curr_df['description']), one_hot_matrix )).toarray()
            curr_df.drop(['description'], axis=1, inplace=True)

            setting_str = ' %-6s outlier_%-4s '%(feature_eng.__name__, outlier_type)

            ###########################################
            ###### SKlearn Regresors
            ########################
            if Exp_options['run_SK_models']:SK_lvl1(x, curr_df.copy(), price_lst, avg_shifting_dic, zero_one_shifting_dic,
                                                setting_str, data_set_name, save_ML_model = save_ML_model)


            #######################################
            ###### H2O Regresors
            ####################
            if Exp_options['run_H2O_models']:H2O_lvl1(x, curr_df.copy(), price_lst, avg_shifting_dic, zero_one_shifting_dic,
                                                setting_str, data_set_name, save_ML_model = save_ML_model)


            #######################################
            ###### Statistical moldes
            #########################
            if Exp_options['run_Statistical_models']:
                Statistical_models(x, 'price_Orig_0.0', curr_df[['price_Orig_0.0', 'Categorical_Feature_LE']].copy(),
                                   data_set_name = data_set_name, save_ML_model = save_ML_model)

        #######################################
        ###### LSTM_Regression
        #############################
        if Exp_options['run_LSTM_model']:LSTM_lvl1(   one_hot_matrix = one_hot_matrix,
													  df = curr_df.copy(),
													  price_lst = price_lst,
													  avg_shifting_dic = avg_shifting_dic,
													  zero_one_scaler_dic = zero_one_scaler_dic,
													  setting_str = 'LSTM outlier_%-4s'%outlier_type,
													  data_set_name = data_set_name,
													  epochs = 2)

except Exception as exc:
    write_to_file(data_set_name, '\n**** Err: ' + get_timestamp() +  ' \n',traceback.format_exc())