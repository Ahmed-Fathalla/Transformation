# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: Feature Engineering methods
"""
import pandas as pd

def get_One_Hot_Encoding(df):
    if type(df) is pd.core.series.Series:
        from keras.utils import np_utils
        One_Hot_product_type = np_utils.to_categorical(df.values.reshape(-1,1))
        return One_Hot_product_type
    else:
        return pd.get_dummies(df, prefix=df.columns,dummy_na = True)

def TF_IDF(dataseries):
    from sklearn.feature_extraction.text import TfidfVectorizer 
    vectorizer = TfidfVectorizer(lowercase= True, max_features = 1000,
                            min_df = 10, max_df = 0.999, use_idf=False)
    
    return vectorizer.fit_transform(dataseries)

def BOW(dataseries):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(lowercase=True,min_df = 10, max_features = 1000,
                                 max_df = 0.999)
    return vectorizer.fit_transform(dataseries)

def get_Embedding_matrix(dataseries, maxlen=300):
    import scipy
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dataseries)
    vocab_size = len(tokenizer.word_index) + 1

    encoded_docs = tokenizer.texts_to_sequences(dataseries)
    return scipy.sparse.csr_matrix(pad_sequences(encoded_docs, maxlen = maxlen, padding='pre')), vocab_size
    
def lvl2_feature_engineering(temp_df, col):
    def up_down_pred(x, col):
        if x[col]>x['avg_shifted']: return 1
        else: return 0
    temp_df['up_down_pred'] = temp_df.apply(lambda x: up_down_pred(x, col), axis=1)
    temp_df['y_pred_transform_divid_avg_1'] = (temp_df[col] - temp_df['avg_shifted']) / temp_df['avg_shifted']
    temp_df['y_pred_transform_divid_avg_2'] = temp_df[col] / temp_df['avg_shifted']
    feature_lst = [
                   col,
                   'avg_shifted',
                   'up_down_pred', 
                   'y_pred_transform_divid_avg_1',
                   'y_pred_transform_divid_avg_2'
                  ]
    return temp_df, feature_lst