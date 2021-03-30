import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import MinMaxScaler


def performance_threshold(df, threshold):
    '''
    input: a cleaned dataframe containing train data and outperform/underperform threshold x%
    output: a cleaned dataframe containing train data with outperform/underperform above x% and below -x%
    '''
    df = df[(df['3-year Annualized Return vs. S&P500']>=threshold) | 
            (df['3-year Annualized Return vs. S&P500']<=-threshold)].reset_index(drop=True)
    
    return df

def distribution_plot(df, column_name):
    '''
    input: a cleaned dataframe containing train data and a column name of the feature we want to see a distribution plot for
    output: a distribution plot showing outperforming and underperforming funds separately
    '''
    plt.figure(figsize=(8,5))
    df[df['Outperform / Underperform']==1][column_name].hist(alpha=0.5,
                                                  color='blue', bins=30,label='Outperform')
    df[df['Outperform / Underperform']==0][column_name].hist(alpha=0.5,
                                                  color='red', bins=30,label='Underperform')
    plt.legend()
    plt.grid(b=None)
    plt.xlabel(column_name)
    plt.title(column_name + ' Distribution', fontsize=15)
    
def count_plot(df, column_name):
    '''
    input: a cleaned dataframe containing train data and a column name of the categorical feature we want to see a count plot for
    output: a count plot of of the categorical feature showing outperforming and underperforming funds separately
    '''
    plt.figure(figsize=(8,5))
    sns.countplot(column_name, data=df, hue='Outperform / Underperform')
    
def count_plot2(df,column_name):
    '''
    input: a cleaned dataframe containing train data and a column name of the categorical feature we want to see a count plot for
    output: a count plot of of the categorical feature showing outperforming and underperforming funds separately
    '''
    plt.figure(figsize=(8,5))
    cp = sns.countplot(column_name, data=df, hue='Outperform / Underperform', palette=['C3','C0'], alpha=0.85)
    legend_labels, _= cp.get_legend_handles_labels()
    cp.legend(legend_labels, ['Underperform','Outperform'])
    cp.set(ylabel=None)
    plt.title(column_name, fontsize=15)

def convert_cat_to_dummy(df, column_name):
    '''
    input: a cleaned dataframe and a column name of the categorical feature we want to convert into dummy variables
    output: a cleaned dataframe with dummy variables
    '''
    dummy = pd.get_dummies(df[column_name], drop_first=True)
    df.drop([column_name], axis=1, inplace=True)
    df = pd.concat([df, dummy], axis=1)
    
    return df

def univ_feature_selection(X, y, stat_test, k):
    '''
    input: a feature set, target, and a statistical test to perform
    output: a dataframe with a boolean value for each feature 
    '''
    if stat_test==chi2:
        scaler_minmax = MinMaxScaler()
        X_scaled = scaler_minmax.fit_transform(X)
        
        sel_test = SelectKBest(stat_test, k=k)
        sel_test.fit_transform(X_scaled, y)
        df = pd.DataFrame(sel_test.get_support(), index=X.columns, columns=['Feature Selection (True/False)'])
        
    else:
        sel_test = SelectKBest(stat_test, k=k)
        sel_test.fit_transform(X, y)
        df = pd.DataFrame(sel_test.get_support(), index=X.columns, columns=['Feature Selection (True/False)'])
    
    return df
        
def rec_feature_selection(X, y, classifer, k):
    '''
    input: a feature set, target, and a classfication model
    output: a dataframe with a boolean value for each feature
    '''
    model = classifer(random_state=100, n_estimators=50)
    sel_test = RFE(estimator=model, n_features_to_select=k, step=1)
    sel_test.fit_transform(X, y)
    df = pd.DataFrame(sel_test.get_support(), index=X.columns, columns=['Feature Selection (True/False)'])
    
    return df

def sfm_feature_selection(X, y, classifier):
    '''
    input: a feature set, target, and a classfication model
    output: a dataframe with a boolean value for each feature
    '''
    model = classifier(random_state=100, n_estimators=50)
    model.fit(X, y)
    sel_model = SelectFromModel(estimator=model, prefit=True, threshold='mean')
    sel_model.transform(X)
    df = pd.DataFrame(sel_model.get_support(), index=X.columns, columns=['Feature Selection (True/False)'])

    return df
