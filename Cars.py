#
from utils import *
data_set_name = 'Cars'# os.path.basename(sys.argv[0]).split('.')[0]

try:
    # Reading the dataframe
    df = pd.read_csv('transformation_ds//%s.csv'%data_set_name, encoding="ISO-8859-1")

    df.rename( columns= {'model':'Categorical_Feature'}, inplace=True )
    df.drop(['seller','dateCrawled','dateCreated','nrOfPictures','lastSeen','postalCode','name','yearOfRegistration',
        'powerPS','monthOfRegistration','notRepairedDamage'], axis=1, inplace=True)

    df = df[ (~df.Categorical_Feature.isnull())]
    df['kilometer'] = df['kilometer'].astype(str)
    df = df[~(df.Categorical_Feature == 'serie_1') & ~(df.Categorical_Feature == 'discovery_sport')]

    lst = ['offerType', 'abtest', 'vehicleType', 'gearbox', 'Categorical_Feature',
               'kilometer', 'fuelType', 'brand']


    for outlier_type in ['None','IQR' ][:1]:

        curr_df, avg_shifting_dic, zero_one_scaler_dic, price_lst = handle_df(df[['Categorical_Feature', 'price']], outlier_type,
                                                                                data_set_name, scaling_lvl2 = True)
        curr_df = pd.merge(df[[*lst, 'price']],
                           curr_df,
                           left_on=df.index, right_on=curr_df.index, how='right')

        x = get_One_Hot_Encoding(curr_df[lst]).values

        curr_df.drop(lst, axis=1, inplace=True)
        setting_str = ' %-6s outlier_%-4s '%('None', outlier_type)

        # print(price_lst)
        # import sys;sys.exit()

        price_lst = price_lst[3:]
        ##########################################
        ###### SKlearn Regression
        #################################
        if Exp_options['run_SK_models']:SK_lvl1(x, curr_df.copy(), price_lst,
                                                avg_shifting_dic, zero_one_scaler_dic,
                                                setting_str, data_set_name, save_ML_model = save_ML_model)

        #######################################
        ###### H2O Regression
        #############################
        # if Exp_options['run_H2O_models']:H2O_lvl1(x, curr_df.copy(), price_lst,
        #                                           avg_shifting_dic, zero_one_scaler_dic,
        #                                           setting_str, data_set_name, save_ML_model = save_ML_model)

        #######################################
        ###### Statistical moldes
        #########################
        # if Exp_options['run_Statistical_models']:
                # Statistical_models(x, 'price_Orig_0.0', curr_df[['price_Orig_0.0', 'Categorical_Feature_LE']].copy(),
                                   # data_set_name = data_set_name, save_ML_model = save_ML_model)

except Exception as exc:
    write_to_file(data_set_name, '\n**** Err: ' + get_timestamp() +  ' \n',traceback.format_exc())










