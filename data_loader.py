# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def load_data(file_name):
    """Load training data and id"""
    path_dataset = file_name
    all_data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1,  usecols= \
                             (0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30) ) #stupid way, see if you manage to change it
                                                                                                                 # (0, 2:30) doesnt work
    char_pred = np.genfromtxt(path_dataset, delimiter=",", dtype = str, skip_header=1, usecols=(1))
    ids = all_data[:,0]
    data = all_data[:,1:]
   
    for i in range(len(char_pred)):
       
       if char_pred[i] == 's':         #change the s into 1s and the b into 0s
            char_pred[i] = int(1)
       else:
            char_pred[i] = int(0)
           
    pred = [int(j) for j in char_pred]   #make an array of integer out of an array of string
    pred = np.array(pred)
    ids = np.array(ids)
    data = np.array(data)
    # predictions = all_data[:,1]  
    # data = all_data[:,2:]
    return ids, data, pred

#def load_test_data():
#    """Load test data and id"""
#    path_dataset = "test.csv"
#    all_data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[0, 31])
#    ids = all_data[:,0]
#    data = all_data[:,2:]
#    return ids, data

def load_training_dataframe():
    whole_dataframe = pd.read_csv ("train.csv")
    DER_dataframe = pd.read_csv ("train.csv", usecols=range(0,15))
    PRI_dataframe = pd.read_csv ("train.csv", usecols=[0,1].append(range(0,15)))
    return whole_dataframe, DER_dataframe, PRI_dataframe
    
def load_test_dataframe():
    whole_dataframe = pd.read_csv ("test.csv")
    DER_dataframe = pd.read_csv ("test.csv", usecols=range(0,15))
    PRI_dataframe = pd.read_csv ("test.csv", usecols=[0,1].append(range(0,15)))
    return whole_dataframe, DER_dataframe, PRI_dataframe