import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt

mlp.rc("figure", max_open_warning=0)


# -----------------------------------------------------------------------------------------------------------


def features_visualization(feature_1, feature_2, train_df, remove_outliers=True):
    """
    Shows scatter plot of 2 given features. points corresponding to a signal are given in red and points
    corresponding to background noise are give in blue

    Args:
        feature_1: string, name of first feature
        feature_2: string, name of second feature
        train_df: pandas dataframe containng the features with their respective names
        remove_outliers: boolean, indicate whether we want to remove undefined values from the scatter plot.
    """
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    new_train_df = train_df
    if remove_outliers:
        new_train_df = train_df[
            np.logical_and(train_df[feature_1] > -999, train_df[feature_2] > -999)
        ]
    ax1.scatter(
        new_train_df[new_train_df["Prediction"] == "s"][feature_1],
        new_train_df[new_train_df["Prediction"] == "s"][feature_2],
        marker=".",
        color=[1, 0.06, 0.06],
        s=0.2,
        label="signal",
    )
    ax1.scatter(
        new_train_df[new_train_df["Prediction"] == "b"][feature_1],
        new_train_df[new_train_df["Prediction"] == "b"][feature_2],
        marker="*",
        color=[0.06, 0.06, 1],
        s=0.2,
        label="background noise",
    )
    ax1.set_xlabel(feature_1)
    ax1.set_ylabel(feature_2)
    ax1.legend(loc='upper right')
    ax1.grid()
    
    
# -----------------------------------------------------------------------------------------------------------

    
def plots_distribution_boxplots(Names, train_df):
    """
    Shows distribution and boxplots of a feature specified by its name in a given list
    Saves images in .png format

    Args:
        Names: list of string, the names of the features we would like to visualize in a
        given dataframe
        train_df: pandas dataframe containng the features with their respective names
    """  
    
    # plot the distributions
    figs, axs = plt.subplots(4,7, figsize=(40,30))
    for i, name in enumerate(Names):
        axis = axs[np.divmod(i, 7)]
        axis.hist(train_df[train_df[name] > -999][name],bins=100)
        axis.set_xlabel(name)
    figs.suptitle('Feature distribution', fontsize=50)
    plt.savefig("imgs/distributions.png")
    
    # plot the boxplots
    figs_2, axs_2 = plt.subplots(4,7, figsize=(40,30))
    for i, name in enumerate(Names):
        axis = axs_2[np.divmod(i, 7)]
        axis.boxplot(train_df[train_df[name] > -999][name])
        axis.set_xlabel(name)
    figs_2.suptitle('Boxplots of features', fontsize=50)
    plt.savefig("imgs/boxplots.png")