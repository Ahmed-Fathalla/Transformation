import time

from statsmodels.formula.api import mixedlm, gee
# from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Exchangeable

from ._scoring_metrics import get_statistical_model_results
from ._utils import get_train_test_indexes
from ._config import get_statistical_model_summary

def mixedlm_(new_df, data_set_name, save_ML_model, get_model_summary = get_statistical_model_summary):

    train_index, test_index = get_train_test_indexes(new_df['Categorical_Feature_LE'])
    x_train = new_df.loc[train_index]
    x_test =  new_df.loc[test_index]

    feat_str = ' + ' .join(x_train.columns[2:])
    md = mixedlm( 'y ~ '+feat_str ,x_train, groups=x_train['Categorical_Feature_LE']  )

    start_time = time.time()
    model = md.fit()
    fitting_time = str( (time.time() - start_time)/60)[:5]

    if save_ML_model:model.save("mixedlm_%s.pickle"%data_set_name)
    model_name = str(model).split('.')[3].split('Result')[0]

    # get and print the model results
    get_statistical_model_results(model, model_name,
                                  x_train,x_test,
                                  feat_str, fitting_time,
                                  data_set_name, get_model_summary)

def GEE_(new_df, data_set_name, save_ML_model, get_model_summary = get_statistical_model_summary):

    train_index, test_index = get_train_test_indexes(new_df['Categorical_Feature_LE'])
    x_train = new_df.loc[train_index]
    x_test =  new_df.loc[test_index]

    feat_str = ' + ' .join(x_train.columns[2:])
    
    # model1 = GEE.from_formula('y ~ '+feat_str, "Categorical_Feature_LE",data=x_train, family=Gaussian(), cov_struct=Exchangeable())
    model1 = gee('y ~ '+feat_str, "Categorical_Feature_LE",data=x_train, family=Gaussian(), cov_struct=Exchangeable())
    
    start_time = time.time()
    model = model1.fit()
    fitting_time = str( (time.time() - start_time)/60)[:5]

    if save_ML_model:model.save("GEE_%s.pickle"%data_set_name)
    model_name = str(model).split('.')[3].split('Result')[0]

    # get and print the model results
    get_statistical_model_results(model, model_name,
                                  x_train,x_test,
                                  feat_str, fitting_time,
                                  data_set_name, get_model_summary)