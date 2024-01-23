# -*- coding:utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data
import platform
from matplotlib import font_manager, rc

path = "c:/Windows/Fonts/malgun.ttf"
if platform.system() == "Windows":
   font_name = font_manager.FontProperties(fname=path).get_name()
   rc("font", family=font_name)
plt.rcParams["axes.unicode_minus"] = False

total_df = load_data()
df = total_df

def plot_target():

    st.write("""### 타겟 데이터 분포""")
    st.write("Note: 타겟 컬럼은 'Exited'")

    df['Exited_label'] = df['Exited'].map({1: 'Exited', 0: 'Not Exited'})
    fig, ax = plt.subplots(1,2,figsize=(15, 6), width_ratios=[2,1])
    textprops={'fontsize': 12, 'weight': 'bold',"color": "black"}
    ax[0].pie(df['Exited_label'].value_counts().to_list(),
            colors=["#abc9ea","#98daa7","#f3aba8","#d3c3f7","#f3f3af","#c0ebe9"],
            labels=df['Exited_label'].value_counts().index.to_list(),
            autopct='%1.f%%', 
            explode=([.05]*df['Exited_label'].nunique()),
            pctdistance=0.5,
            wedgeprops={'linewidth' : 1, 'edgecolor' : 'black'}, 
            textprops=textprops)

    sns.countplot(x = 'Exited_label', data=df, palette = "pastel6", order=df['Exited_label'].value_counts().to_dict().keys())
    for p, count in enumerate(df['Exited_label'].value_counts()):
        ax[1].text(p-0.11, count+np.sqrt(count)+1000, count, color='black', fontsize=13)
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    plt.yticks([])
    plt.box(False)

    fig.suptitle(x=0.56, t=f"이탈 컬럼 분포", fontsize=18, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

def cat_cols_graph():
    st.write("""### 범주형 데이터 분포""")
    st.write("Note: 범주형 컬럼은 'Geography', 'Gender', 'HasCrCard', 'IsActiveMember'")

    cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    for column in cat_cols:
        fig , ax = plt.subplots(1,2,figsize=(18,5.5))
        df[column].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True,colors=["#abc9ea","#98daa7","#f3aba8","#d3c3f7","#f3f3af","#c0ebe9"])
        ax[0].set_ylabel(f'{column}')
        sns.countplot(x=column,data=df,ax=ax[1], palette=["#abc9ea","#98daa7","#f3aba8","#d3c3f7","#f3f3af","#c0ebe9"])
        ax[1].bar_label(ax[1].containers[0])
        plt.suptitle(f'{column}')
        st.pyplot(fig)

# def cat_exited_graph():
#     st.write("""### 이탈 여부에 따른 범주형 데이터 분포""")

def cat_exited_graph():
    st.write("""### 이탈 여부에 따른 범주형 데이터 분포""")
    global df
    
    # geography
    geo_Exited_counts = (
        df.groupby(["Geography", "Exited"]).size().unstack(fill_value=0).reset_index()
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=geo_Exited_counts["Geography"],
                y=geo_Exited_counts[1],
                name="Exiteded",
                marker_color="#f3aba8",
            ),
            go.Bar(
                x=geo_Exited_counts["Geography"],
                y=geo_Exited_counts[0],
                name="Not Exiteded",
                marker_color="#abc9ea",
            ),
        ]
    )
    fig.update_layout(
        title="지역에 따른 이탈 여부 분포".center(150),
        xaxis_title="Geography",
        yaxis_title="Count",
        barmode="group",  # To create grouped bars
    )
    st.plotly_chart(fig)

    # gender
    geo_Exited_counts = (
        df.groupby(["Gender", "Exited"]).size().unstack(fill_value=0).reset_index()
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=geo_Exited_counts["Gender"],
                y=geo_Exited_counts[1],
                name="Exiteded",
                marker_color="#f3aba8",
            ),
            go.Bar(
                x=geo_Exited_counts["Gender"],
                y=geo_Exited_counts[0],
                name="Not Exiteded",
                marker_color="#abc9ea",
            ),
        ]
    )
    fig.update_layout(
        title="성별에 따른 이탈 여부 분포".center(150),
        xaxis_title="Gender",
        yaxis_title="Count",
        barmode="group",  
    )
    st.plotly_chart(fig)

    # HasCrCard
    geo_Exited_counts = (
        df.groupby(["HasCrCard", "Exited"]).size().unstack(fill_value=0).reset_index()
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=geo_Exited_counts["HasCrCard"],
                y=geo_Exited_counts[1],
                name="Exiteded",
                marker_color="#f3aba8",
            ),
            go.Bar(
                x=geo_Exited_counts["HasCrCard"],
                y=geo_Exited_counts[0],
                name="Not Exiteded",
                marker_color="#abc9ea",
            ),
        ]
    )
    fig.update_layout(
        title="신용카드 유무에 따른 이탈 여부 분포".center(150),
        xaxis_title="HasCrCard",
        yaxis_title="Count",
        barmode="group",  
    )
    st.plotly_chart(fig)

    # IsActiveMember
    geo_Exited_counts = (
        df.groupby(["IsActiveMember", "Exited"]).size().unstack(fill_value=0).reset_index()
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=geo_Exited_counts["IsActiveMember"],
                y=geo_Exited_counts[1],
                name="Exiteded",
                marker_color="#f3aba8",
            ),
            go.Bar(
                x=geo_Exited_counts["IsActiveMember"],
                y=geo_Exited_counts[0],
                name="Not Exiteded",
                marker_color="#abc9ea",
            ),
        ]
    )
    fig.update_layout(
        title="활성멤버 여부에 따른 이탈 여부 분포".center(150),
        xaxis_title="IsActiveMember",
        yaxis_title="Count",
        barmode="group",  
    )
    st.plotly_chart(fig)


def num_exited_graph():
    st.write("""### 이탈 여부에 따른 수치형 데이터 분포""")
    st.write("Note: 수치형 컬럼은 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'")

    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']  
    for column in num_cols:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.histplot(x=column, hue='Exited', data=df, kde=True, bins=25, palette=["#abc9ea","#98daa7","#f3aba8","#d3c3f7","#f3f3af","#c0ebe9"])
        st.pyplot(fig)

def age_to_cat_graph():
    st.write("Note: 수치형 컬럼인 'Age' 데이터를 연령대별로 나눈 상세 분포")

    df['age_to_cat'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 100], labels=['18-30', '30-40', '40-50', '50-60', '60+'])
    colors = ["#abc9ea","#98daa7","#f3aba8","#d3c3f7","#f3f3af","#c0ebe9"]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.countplot(x='age_to_cat', hue='Exited', data=df, palette=colors)
    ax.bar_label(ax.containers[0], label_type='edge', fontsize=11)
    ax.bar_label(ax.containers[1], label_type='edge', fontsize=11)
    st.pyplot(fig)

    # 파이차트
    age_category_counts = df["age_to_cat"].value_counts().reset_index()
    age_category_counts.columns = ["age_to_cat", "Count"]

    fig = px.pie(
        age_category_counts,
        values="Count",
        names="age_to_cat",
        title="".center(160),
        color_discrete_sequence=colors,
        hole=0.2,
    )
    # Add labels inside the pie chart
    fig.update_traces(
        textinfo="percent+label",
        textfont_size=14,
        pull=[0.08, 0.08, 0.08, 0.08]
    )

    # Customize the layout and appearance
    fig.update_layout(
        legend=dict(
            x=0.85,
            y=0.5,  # Adjust the legend position
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="gray",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, b=0, t=30), 
    )

    st.plotly_chart(fig)


def cat_heatmap():
    st.write("""### 범주형 데이터의 상관계수""")
    df_cat = df.select_dtypes(exclude=['int64', 'float64']).drop(['Surname', 'Exited_label', 'age_to_cat'], axis=1)
    df_cat = df_cat.apply(lambda x: pd.factorize(x)[0])
    df_cat_corr = pd.concat([df_cat, df[['HasCrCard', 'IsActiveMember', 'Exited']]], axis=1).corr()
    fig = px.imshow(df_cat_corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale='RdBu', aspect='auto')
    st.plotly_chart(fig)
    

def num_heatmap():
    st.write("""### 수치형 데이터의 상관계수""")
    df_digit = df.select_dtypes(include=['int64', 'float64']).drop(['id', 'CustomerId'], axis=1)
    df_digit.drop(['HasCrCard', 'IsActiveMember'], axis=1, inplace=True)
    fig = px.imshow(df_digit.corr(), text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale='RdBu', aspect='auto')
    st.plotly_chart(fig)