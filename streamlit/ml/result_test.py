# -*- coding:utf-8 -*-

import streamlit as st
import pandas as pd
from utils import load_test_data, load_models, process_test_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def get_clf_eval(y_test, pred, pred_proba):
    acc = accuracy_score(y_test,pred)
    pre = precision_score(y_test,pred)
    re = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    auc = roc_auc_score(y_test,pred_proba)    
    return acc, pre, re, f1, auc

def plot_confusion_matrix(confusion):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues', linewidths=0.8)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')    
    st.pyplot(plt)


def print_clf_eval(y_test, pred, pred_proba):
    confusion = confusion_matrix(y_test, pred)
    acc, pre, re, f1, auc = get_clf_eval(y_test, pred, pred_proba)

    st.markdown("#### Confusion Matrix")
    plot_confusion_matrix(confusion)
    result = pd.DataFrame({'Accuracy': [acc], 'Precision': [pre], 'Recall': [re], 'F1': [f1], 'ROC_AUC': [auc]})
    st.markdown("##### 평가 지표")
    st.dataframe(result, use_container_width=True)

    
def run_test():
    df_test = load_test_data()
    models, names = load_models()
    X, y, customers = process_test_data(df_test)

    st.markdown("##### 테스트 데이터\n"
                '- 데이터 포인트 : 10,002개')
    st.markdown('<hr>', unsafe_allow_html=True)
    
    for model in models:        
        if isinstance(model.estimator, CatBoostClassifier):
            cat = model.best_estimator_
            cat_pred = cat.predict(X)
            cat_proba = cat.predict_proba(X)[:,1]
            
            st.markdown("### Predictions by Catboost:")
            print_clf_eval(y, cat_pred, cat_proba)
            
        elif isinstance(model.estimator, XGBClassifier):
            xgb = model.best_estimator_            
            xgb_pred = xgb.predict(X)
            xgb_proba = xgb.predict_proba(X)[:,1]
            
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown("### Predictions by XGBoost:")
            print_clf_eval(y, xgb_pred, xgb_proba)
        else:
            pass
