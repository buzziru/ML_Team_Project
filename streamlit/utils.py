# -*- coding:utf-8 -*-

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import os
import joblib


@st.cache_data
def load_data():
    data = pd.read_csv('data/train.csv/train.csv')
    return data


@st.cache_data
def load_test_data():
    data = pd.read_csv('data/bank_turnover/Churn_Modelling.csv')
    return data


@st.cache_data
def do_preprocess(df):
    df = df.drop(columns=['id', 'Surname'])
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    customers = df['CustomerId']
    df = df.drop(columns=['CustomerId'])
    ndf = pd.get_dummies(df, columns=['Geography','Gender'])

    X = ndf.drop(['Exited'], axis=1)
    y = ndf['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    return X_test, y_test, customers


@st.cache_data
def process_test_data(df):
    df = df.drop(columns=['RowNumber', 'Surname'])
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    customers = df['CustomerId']
    df_test = df.drop(columns=['CustomerId'])
    ndf = pd.get_dummies(df_test, columns=['Geography','Gender'])

    X = ndf.drop(['Exited'], axis=1)
    y = ndf['Exited']
    return X, y, customers


@st.cache_resource
def load_models():
    models = []
    names = []
    
    model_names_map = {
        'cat': 'CatBoost',
        'lr': 'Logistic Regression',
        'dt': 'Decision Tree',
        'lgbm': 'LightGBM',
        'xgb': 'XGBoost',
        'rf': 'Random Forest'
    }
    for filename in os.listdir('streamlit/ml/models'):
        if filename.endswith('_gs.joblib'):
            model_path = os.path.join('./ml/models', filename)
            loaded_model = joblib.load(model_path)
            
            name = filename.split('_gs')[0]
            model_name = model_names_map.get(name, 'Unknown')
            
            models.append(loaded_model)
            names.append(model_name)
    return models, names
