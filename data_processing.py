import numpy as np
import pandas as pd

def one_hot_encoding(df, number_of_bins, name='PRI_jet_num'):
    for i in range(int(number_of_bins-1)):
        df[name+"encoding_{n}".format(n=i)] = np.where(df[name]==i,1,0) 
    if (name=='PRI_jet_num'):
        df["PRI_jet_num_encoding_<=1"] = np.where(np.logical_and(df['PRI_jet_num']==0,df['PRI_jet_num']==1),1,0)
        df["DER_mass_MMC_encoding_undefined"] = np.where(df["DER_mass_MMC"]==-999, 1,0)
    df = df.drop(name, axis=1)
    
#def binning(df,Names,number_of_bins=4):
    #for name in Names:
        #df[name] = pd.qcut(df[name], number_of_bins, labels=False)
        #one_hot_encoding(df,name,number_of_bins)
        
def handle_outliers(df,Names):
    for name in Names:
        df_test = df[df[name]>-999][name]
        median = df_test.median()
        IQR= df_test.quantile(0.75) - df_test.quantile(0.25)
        lower_bridge= df_test.quantile(0.25)-(IQR*3)
        upper_bridge= df_test.quantile(0.75)+(IQR*3)
        df[df[name]>-999][name] = np.where(np.logical_or(df_test<lower_bridge,df_test>upper_bridge),df_test,median)
    
def handle_undefined_values(df,Names):
    for name in Names:
        median = df[df[name]>-999][name].median()
        df[df[name]==-999][name] = median
        if (name == 'PRI_jet_all_pt'):
            df[df[name]==0][name] = df[df[name]==0][name].median()

def handle_skew(df,Names):
    for name in Names:
        df_test = df[df[name]>-999][name]
        df[df[name]>-999][name] = np.log(df_test)
    
def standardize(df):
    for name in list(df.drop(['Id','Prediction'], axis=1)):
        if ('encoding' not in name):
            df[name] = (df[name]-df[name].mean())/df[name].var()