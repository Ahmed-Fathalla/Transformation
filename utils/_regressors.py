# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: Sklearn and H2O regression Algorithms used in the first and second level regression
"""

from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator 

from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


SK_lvl1_model_dic = {
    'Amazon':       (RandomForestRegressor(n_estimators=30, max_depth=13, n_jobs = -1, random_state =1),'SK_RF_30_13'),
    'Mercari':      (RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs = -1, random_state =1),'SK_RF_20_10'),
    'Inside_Airbnb':(RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs = -1, random_state =1),'SK_RF_20_10'),
    'Cars':         (RandomForestRegressor(n_estimators=30, max_depth=10, n_jobs = -1, random_state =1),'SK_RF_30_10'),
    }

SK_model_lst = [
                (LinearSVR(fit_intercept=False, random_state=1),'SK_LinearSVR'),
                (LinearRegression(fit_intercept=False, n_jobs=-1),'SK_LR'),
                (MLPRegressor(hidden_layer_sizes=(512,128),
                              activation='relu',
                              solver='adam',
                              learning_rate='adaptive',
                              max_iter=1000,learning_rate_init=0.01,alpha=0.01, random_state=1),'SK_MLP'),
                (GradientBoostingRegressor(random_state=1),'SK_GBRegressor'),
                (RandomForestRegressor(n_estimators=30, max_depth=13, n_jobs = -1, random_state =1),'SK_RF_30_13')
      ]
              
H2O_model_lst = [
                 H2OGeneralizedLinearEstimator(family= "gaussian", compute_p_values = True,remove_collinear_columns = True, lambda_ = 0,seed=1),
                 H2OGradientBoostingEstimator(seed=1),
                 H2OXGBoostEstimator(seed = 1),
                 H2ORandomForestEstimator(ntrees=30, max_depth=13,seed=1),
            ]


