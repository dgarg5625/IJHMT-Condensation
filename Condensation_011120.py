

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBRegressor

from glob import glob
from os.path import join, abspath
from os import listdir, getcwd

import RFF
import MLP
import RandFor
import AdaBoost
import XgBoost

if __name__ == "__main__":
    
    # Read all the .xls files. Files saves with .xlsx will not be read. 
    path = os.getcwd()
    print(path)
    
    
    # files = os.listdir(path)
    
    # files_xls = [f for f in files if f[-3:] == 'xls']
    # print(len(files_xls))
    
    # # Concatenate all the xl files into 1 dataframe. 
    # df_ALL = pd.DataFrame()
    # for f in files_xls:
    #     data = pd.read_excel(f, 'Sheet1')
    #     data['filename'] = os.path.basename(f)
    #     df_ALL = df_ALL.append(data)
        
    # df_ALL = df_ALL.iloc[2:,:]
    # df_ALL.rename(columns={'temperature.1': 'Temperature'}, inplace=True)
    # df_consol = df_ALL
    # df_consol = df_consol.loc[df_consol.index >1]
    
    
    ############ TO READ EXCEL FILES FROM BRUCE FOLDER################
    path1 = os.path.join(path, "For Bruce")
    files = os.listdir(path1)

    files_xls1 = [f for f in files if f[-3:] == 'xls']
    print(len(files_xls1))

    # Concatenate all the xl files into 1 dataframe. 
    # df_ALL1 = pd.DataFrame()
    # full_path = join(path1, "*.xls")
    # for f in glob(full_path):
    #     data = pd.read_excel(f, 'Sheet1')
    #     data['filename'] = os.path.basename
    #     df_ALL1 = df_ALL1.append(data)
        
    df_ALL1 = pd.DataFrame()
    for f in files_xls1:
        data = pd.read_excel( os.path.join(path1, f ), 'Sheet1')
        data['filename'] = os.path.basename(f)
        df_ALL1 = df_ALL1.append(data)   
        
    df_ALL1 = df_ALL1.iloc[2:,:]
    df_ALL1.rename(columns={'temperature.1': 'Temperature'}, inplace=True)
    df_consol_b = df_ALL1
    df_consol_b = df_consol_b.loc[df_consol_b.index >1]
    
    df_consol = df_consol_b.copy()
    ################### DELETE ABOVE LINES###############################
    
    # Get all column names
    #print(df_consol.columns)
    
    drop_cols = ['Bo','HT Chracteristic', 'HT characteristic', 'HT characteristics']
    
    df_consol1 = df_consol.drop(drop_cols,axis=1)
    df_consol1.fillna(0,inplace=True)
    
    # df_Author = df_consol1['filename'].value_counts().to_frame()
    # df_Author.reset_index(inplace=True)
    # df_Author.head(2)
    
    # df_consol1['filename']=df_consol1['filename'].astype('category')
    
    # df_consol1['filename']=df_consol1['filename'].cat.codes
    # df_Author_num = df_consol1['filename'].value_counts().to_frame()
    # df_Author_num.reset_index(inplace=True)   
    # df_Author_map = df_Author.merge(df_Author_num, how='outer', left_index=True, right_index=True)
    
    #predictors = ['Bd','Co','Cpf','Cpg','Critical Pressure','Cvf','Cvg','Density-f','Density-g','Frf','Frfo','Frg','Frgo','Ga','Ka','PF','PH','PR','Pressure','Prf','Prg','Ref','Refo','Reg','Rego','Suf','Sug','Sufo','Sugo','Wef','Wefo','Weg','Wego','Xtt','Xtv','Xvv','aspect ratio','diameter','hfg','kf','kg','mass velocity','quality','Temperature','μf','μg','σ']
    
    #predictors = ['Bd','Co','Cpf','Cpg','Critical Pressure','Cvf','Cvg','Density-f','Density-g','Frf','Frfo','Frg'
     #              ,'Frgo','Ga','Ka','PF','PH','PR','Pef','Peg','Pressure','Prf','Prg','Ref','Refo','Reg','Rego','Suf'
      #            ,'Sug','Wef','Wefo','Weg','Wego','Xtt','Xtv','Xvt','Xvv','aspect ratio','diameter','hfg','kf'
       #            ,'kg','mass velocity','quality','Temperature','μf','μg','σ']
    
    predictors = ['Bd','Co','Frf','Frfo','Frg','Frgo','Ga','Ka','Prf','Prg','Ref','Refo','Reg','Rego','Suf','Sug','Sufo','Sugo','Wef','Wefo','Weg','Wego']
    #predictors = ['Cpf','Cpg','Critical Pressure','Cvf','Cvg','Density-f','Density-g','PF','PH','PR','Pressure','aspect ratio','diameter','hfg','kf','kg','mass velocity','quality','Temperature','μf','μg','σ']
    
    #predictors = ['Bd','Prf','Ref', 'Frf']
    #predictors = ['diameter','mass velocity','quality']
    #predictors = ['Frg','Frfo','Critical Pressure','Prg','Prf','Cvg']
 
    X_1 = df_consol1[predictors]
    df_tar = df_consol[['h','htp','hann','filename']]
    
    # Split datasetvps into train and test. Random_state is an arbitrary number to reproduce the results. 
    X = X_1
    y = df_tar
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,random_state=10)
    # To calculate Adjusted R2
    p = len(predictors)
    
    # Data to be tested separately
    path1 = os.path.join(path1, "Excluded")    # (path1, "Excluded") modified to path1
    files1 = os.listdir(path1)
    files_xls1 = [f for f in files1 if f[-3:] == 'xls']
    print('DataSet Tested Separately',files_xls1)
    
    
    # Concatenate all the xl files into 1 dataframe. 
    df_ALL1 = pd.DataFrame()
    for f in files_xls1:
        data = pd.read_excel( os.path.join(path1, f ), 'Sheet1')
        data['filename'] = os.path.basename(f)
        df_ALL1 = df_ALL1.append(data)
    df_ALL1 = df_ALL1.iloc[2:,:]
    
    df_ALL1.rename(columns={'temperature.1': 'Temperature'}, inplace=True)
    df_consol1 = df_ALL1
    df_consol1 = df_consol1.loc[df_consol1.index >1]
    df_consol2a = df_consol1[predictors] # cols -> predictors
    
    df_consol2a.fillna(0,inplace=True)
    df_consol2a.head(3)
    
    X_1a = df_consol2a
    df_tar_a = df_consol1[['h','htp','hann','filename']]
            
    
    ##RFF.RFF_regr(X_train,y_train,X_test,y_test)
    
    #RandFor.RandFor(X_train,y_train,X_test,y_test,X_1a,df_tar,df_tar_a,p)
    
    MLP.MLP_regr(X_train,y_train,X_test,y_test,X_1a,df_tar,df_tar_a,p)
    
    #AdaBoost.AdaBoost(X_train,y_train,X_test,y_test,X_1a,df_tar,df_tar_a,p)
    
    #XgBoost.xgboost(X_train,y_train,X_test,y_test,X_1a,df_tar,df_tar_a,p)
    
    

