import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def features_distribution(Names, train_df):
    for name in Names:
        fig = plt.figure(figsize=(8, 4), dpi=80)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1 = train_df[train_df[name]> -999][name].hist(bins=100)
        ax1.set_xlabel(name)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2 = train_df[train_df[name]> -999].boxplot(column=name)
            
def features_visualization(feature_1, feature_2, train_df, remove_outliers=True):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    new_train_df=train_df
    if remove_outliers:
        new_train_df=train_df[np.logical_and(train_df[feature_1]>-999,train_df[feature_2]>-999)]
    ax1.scatter(
        new_train_df[new_train_df['Prediction'] =='s'][feature_1], 
        new_train_df[new_train_df['Prediction'] =='s'][feature_2],
        marker='.', color=[1, 0.06, 0.06], s=0.2, label="signal")
    ax1.scatter(
        new_train_df[new_train_df['Prediction'] =='b'][feature_1],
        new_train_df[new_train_df['Prediction'] =='b'][feature_2],
        marker='*', color=[0.06, 0.06, 1], s=0.2, label="background noise")
    ax1.set_xlabel(feature_1)
    ax1.set_ylabel(feature_2)
    ax1.legend()
    ax1.grid()
            