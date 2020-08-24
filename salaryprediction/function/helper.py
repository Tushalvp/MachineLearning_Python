#!/usr/bin/env python
# coding: utf-8

# In[54]:

# Import Python Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
import math
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

# Function to load csv file from a centralised path
def load_csv(project,filename):
    """load data from csv file and creates dataframe"""
    rawdatapath = "D:/ML_Portfolio/Development/data/raw/"
    df = pd.read_csv("{0}/{1}/{2}.csv". format(rawdatapath,project,filename))    
    return df

# Function to identify the datatype, shape, missing values and dupcliate values
def data_explore(df):
    print('Data Shape : {}\n'.format(df.shape))
    print('------------------------------')
    print('Duplicate Values....')
    print('------------------------------')
    print('{}\n'.format(df.duplicated().value_counts()))
    print('------------------------------')
    print('Null Values....')
    print('------------------------------')
    print('{}\n'.format(df.isnull().sum()))
    print('------------------------------')
    print('Data type....')
    print('------------------------------')
    print('{}\n'.format(df.info()))
# Detects outlier using IQR range
def detect_outlier(df,cols):
    '''finds lower limt and upper limit using inter-quartile range'''
    subset = df[cols].describe()
    print('=============================')
    print(subset)    
    IQR = subset['75%'] - subset['25%']
    upper_limit = subset['75%'] + 1.5 * IQR 
    lower_limit = subset['25%'] - 1.5 * IQR 
    print('-----------------------------')
    print('Lower limit: {}\nUpper limit: {}'.format(lower_limit,upper_limit))
    return lower_limit, upper_limit    

# Returns descriptive statistics by aggregating target variable 
def FeatureEng_GrpAvg(df,cat_cols,target_col):
    '''Returns descriptive statistics by aggregating target variable '''
    groups = df.groupby(cat_cols)
    group_stats_df = pd.DataFrame({'group_mean': groups[target_col].mean()})
    group_stats_df['group_max'] = groups[target_col].max()
    group_stats_df['group_min'] = groups[target_col].min()
    group_stats_df['group_std'] = groups[target_col].std()
    group_stats_df['group_median'] = groups[target_col].median()
    group_stats_cols = group_stats_df.columns.tolist()
    return group_stats_df, group_stats_cols

# Function to merge engineered features to train/test dataframe. Nan values are converted to Zero
def FeatureEng_merge(df, new_df, cat_cols, group_stats_cols, fillna = False):
    ''' Merges the dataframe of the new engineered features with train/test dataframe '''
    new_ds = pd.merge(df, new_df, on = cat_cols, how = 'left')
    if fillna:
        for col in group_stats_cols:
            new_ds[col] = [0 if math.isnan(x) else x for x in new_ds[col]]
    return new_ds

def LabelEncode(df_ref, df_follow, col):
    ''' Label encodes the categorical column of the dataframe, to be able to feed the dataframe to the ML algorithms '''
    le = LabelEncoder()
    le.fit(df_ref[col])
    df_ref[col] = le.transform(df_ref[col])
    df_follow[col] = le.transform(df_follow[col])
    return df_ref[col], df_follow[col], le   

def shuffle_df(df):
    ''' Shuffles the dataframe '''
    return shuffle(df).reset_index()


def cross_val_model(model, feature_df, target_col, n_procs, mean_mse, cv_std):
    ''' Cross validates the model for 50% (cv=2) of the training set'''
    neg_mse = cross_val_score(model, feature_df, target_col, cv = 2, n_jobs = n_procs, scoring = 'neg_mean_squared_error')
    mean_mse[model] = -1.0 * np.mean(neg_mse)
    cv_std[model] = np.std(neg_mse)

def print_summary(model, mean_mse, cv_std):
    ''' Prints a short summary of the model performance '''
    print('\nmodel:\n', model)
    print('Average MSE:\n', mean_mse[model])
    print('Standard deviation during cross validation:\n', cv_std[model])

def get_model_feature_importances(model, feature_df):
    ''' Gets and sorts the importance of every feature as a predictor of the target '''
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = [0] * len(feature_df.columns)
    
    feature_importances = pd.DataFrame({'feature': feature_df.columns, 'importance': importances})
    feature_importances.sort_values(by = 'importance', ascending = False, inplace = True)
    ''' set the index to 'feature' '''
    feature_importances.set_index('feature', inplace = True, drop = True)
    return feature_importances


def compare_catFeature_numFeature(df, cat_var1, cat_var2, target_var):
    '''Creates side-by-side boxplot for categorical features'''
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    sns.boxplot(x=cat_var1, y=target_var, data = df.sort_values(target_var, ascending = True))  
    plt.title(cat_var1 + ' (vs) ' + target_var)
    plt.xticks(rotation=45)
    plt.subplot(1,2,2)
    sns.boxplot(x=cat_var2, y=target_var, data = df.sort_values(target_var, ascending = True))  
    plt.title(cat_var2 + ' (vs) ' + target_var)
    plt.xticks(rotation=45)


def groupmean_heatmap(df,cols):
    '''performs groupby on target variable and creates a corr matrix'''
    for i in cols:
        df[i + ' mean'] = df.groupby([i])['salary'].transform('mean')   # calculated Avg.salary for all cat_cols
    # lets create a heatmap
    df = df.loc[:,df.columns.str.contains("mean|salary|yearsExperience|milesFromMetropolis")] #selecting columns contains 'mean' + Num_cols
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(),cmap="PuBu", annot=True)
    plt.xticks(rotation=45)    