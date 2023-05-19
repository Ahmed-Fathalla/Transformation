
import pandas as pd
from scipy.sparse import hstack
import numpy as np
import time

from ._time_utils import get_timestamp
from ._write_to_file import write_to_file
from ._scoring_metrics import MAE_sklearn, get_score_string

avg_shiftting_factor_lst = ['Orig', 0.0, 0.0001,  0.5]

def check_array(lst):
    '''
    return 1 if the list is sorted, 0 otherwise
    '''
    for i in range(len(lst)-1):
        if not lst[i+1] > lst[i]:
            return False
    return True

def get_slope_intercept(avg_lst, slope_precentage, data_set_name):

    # if avg_lst is not sorted, then raise error. because the Categorical_Feature_LE must be aligned in with avg_lst
    assert check_array(avg_lst), '\navg_lst is not sorted. Check \"get_slope_intercept\"'

    slope = ( avg_lst[-1] - avg_lst[0] ) * slope_precentage
    intercept = avg_lst[0] - slope
    shift_str = 'slope_intercept_str: slope_precentage:%-10s  slop:%-8s  intercept:%-8s\n'%(str(slope_precentage)[:9], str('%-.6f'%slope)[:8] , str(intercept)[:8] )
    return slope, intercept, shift_str

def calculate_residuals(avg_lst, slope, intercept):
    shifts = []
    for i, avg in enumerate(avg_lst):
        y_predicted = slope*(i+1) + intercept
        shifts.append(y_predicted - avg)

    if min(shifts) < 0.0:
        shifts = [min(shifts)*-1 + i for i in shifts]

    return shifts

def get_shifting_values(df, data_set_name):
    avg_lst = df['avg_true'].values
    assert check_array(avg_lst), '\navg_lst is not sorted. Check \"get_shifting_values\"'

    df = df.set_index('Categorical_Feature_LE')
    avg_shifting_factor_str_lst = [str(avg_shiftting_factor) for avg_shiftting_factor in avg_shiftting_factor_lst]

    shfting_str = ''
    for avg_shiftting_factor_str, avg_shiftting_factor in zip(avg_shifting_factor_str_lst, avg_shiftting_factor_lst):
        if avg_shiftting_factor_str == 'Orig':
            df[avg_shiftting_factor_str] = 0
            shfting_str += 'slope_intercept_str: slope_precentage:%-10s  slop:%-8s  intercept:%-8s\n'%( str('Orig')[:5], str('XXXX')[:8] , str('XXXX')[:8] )
        else:
            slope, intercept, shfting_str_ = get_slope_intercept(avg_lst, avg_shiftting_factor, data_set_name)# replace with slope percentage
            shifts = calculate_residuals(avg_lst, slope, intercept)
            df[avg_shiftting_factor_str] = shifts
            shfting_str += shfting_str_

    # saving avg_shifting_dataframe to csv file
    avg_shifting_file = 'csv/avg_shifting_dic_%s_%s.csv'%(data_set_name, get_timestamp())
    df.to_csv(avg_shifting_file)

    write_to_file(data_set_name,'\n','='*(34+len(avg_shifting_file)), '\navg_shifting csv file: %s'%avg_shifting_file,'\n', '='*(23+len(avg_shifting_file)),'\n', shfting_str )
    df.drop(['avg_true'], axis=1, inplace=True)

    return df.to_dict()

def apply_shiftting(df, data_set_name):
    avg_shifting_dic = get_shifting_values(df[['Categorical_Feature_LE','avg_true']].drop_duplicates().sort_values(by='avg_true'), data_set_name)

    if 'Orig' in avg_shifting_dic.keys():
        avg_shifting_factor_str_lst = ['Orig', *sorted(avg_shifting_dic.keys())[:-1]]
    else:
        avg_shifting_factor_str_lst = sorted(avg_shifting_dic.keys())

    col_name_lst = []
    write_to_file(data_set_name,'\n'+'='*80+'\nShifting Average: new price ranges per slop-factor\n'+'='*50)
    for avg_shifting_factor_str in avg_shifting_factor_str_lst:

        ######################################################################
        ###### Applying Mean_shiftting
        ########## ###################
        col_name_lst.append( 'price_'+avg_shifting_factor_str )

        def shifting(row):
            return row['price_scaled'] + avg_shifting_dic[avg_shifting_factor_str][row['Categorical_Feature_LE']]

        s = '%-8s Before: [%-.3f, %-.2f]'%(avg_shifting_factor_str, df['price_scaled'].min(),df['price_scaled'].max() )
        df[col_name_lst[-1]] = df.apply(lambda x: shifting(x), axis=1)
        s += '   After: [%-.5f, %-.5f]'%(df[col_name_lst[-1]].min(),df[col_name_lst[-1]].max() )
        write_to_file(data_set_name,s)

    write_to_file(data_set_name,'-'*80+'\n')
    return df, avg_shifting_dic, col_name_lst

def Back_shift(df, zero_one_scaler, shifting_dict, setting_str_, factor = 0.7, group_factor=2.5):
    str_ = ''
    f = 'NO_zero_one_scaling'
    df['y_pred_orig'] = df['y_pred']
    if zero_one_scaler is not None:
        f = 'zero_one_scaling'
        str_ += '\t$*** %-15s[%-.5f, %-.5f]'%('Before B-scale',min(df['y_pred']),max(df['y_pred']))

        # model_prediction_01_inverse_scaling
        df['y_pred'] = zero_one_scaler.inverse_transform(df['y_pred'].values.reshape(-1,1))

        str_ += '  %-15s[%-.5f, %-.5f]'%('After  B-scale',df['y_pred'].min(),df['y_pred'].max()) + '\n'

    def Back_shifting(row):
        return row['y_pred'] + -1*shifting_dict[row['Categorical_Feature_LE']]

    mae = MAE_sklearn(y_true = df['y_shift'], y_pred = df['y_pred'] )
    # str_ += ' *** mae = ' + str(mae)[:6]
    df['B_shift'] = df.apply(lambda x: Back_shifting(x), axis=1)
    str_ += '    %-22s'%'    $Back_shift' + get_score_string(y_true=df['y_true'],y_pred=df['B_shift']) + '\n\t'

    start_time  = time.time()
    temp_df = df[['y_true','avg_true','B_shift']]
    current_mae = 1
    best_factor = -10

    for x in np.arange(0.1, 1.1, 0.1):
        temp_df['error_value'] = np.where(temp_df['y_true'] < temp_df['avg_true'] , - mae*(x), mae*(x))
        temp_df['final_predicited_value'] = temp_df['B_shift'] + temp_df['error_value']
        temp_mae = MAE_sklearn(y_true = temp_df['y_true'], y_pred = temp_df['final_predicited_value'] )
        # print(x, temp_mae )
        if temp_mae < current_mae:
            current_mae = temp_mae
            best_factor = x
            df['final_predicited_value'] = temp_df['final_predicited_value']
            
    if best_factor == -10:
        df['final_predicited_value'] = df['B_shift']

    elapsed_time  = str( (time.time() - start_time)/60)[:5]
    str_ += '$*** Best_mae = (MAE:' + str(current_mae)[:6] + ', Factor:' + str(best_factor)[:3] + ', time:' + elapsed_time + ')'

    ###########################
    # Method_2
    ###########################
    df['MAE'] = df['y_true'] - df['B_shift']
    from ._transformation import get_mean_values
    df['MAE_per_group'] = np.absolute( get_mean_values(df, 'MAE'))
    df['error_value_group'] = np.where(df['y_true'] < df['avg_true'] , - df['MAE_per_group']*(group_factor), df['MAE_per_group']*(group_factor))
    df['final_predicited_value_groups']   = df['B_shift'] + df['error_value_group']

    # saving B-shift df
    if '/' in setting_str_:
        setting_str_ = 'lvl_2 ' + setting_str_.split('^')[0] + ' ' + setting_str_.split('/')[2]
    else:
        pass
    df.to_csv('csv/B_shift %s %s'%(setting_str_,f)+'.csv')
    return df['final_predicited_value'], df['final_predicited_value_groups'], str_

