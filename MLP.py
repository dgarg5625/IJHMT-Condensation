# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:25:01 2020

@author: Deepak Garg
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import *
import xlsxwriter as xlsw

def MLP_regr(X_train,y_train,X_test,y_test,X_1a,df_tar,df_tar_a,p):
    
    scaler = StandardScaler().fit(X_train)
    scaler.fit(X_train)
    X_train1 = scaler.transform(X_train)
    X_test1 = scaler.transform(X_test)
    
    mlp = MLPRegressor(activation='relu', alpha=0.0001, batch_size=200, beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(150,140,130,120,110,100,90,80,70,60,50,40,30,20,10),
       learning_rate='constant', learning_rate_init=0.001, max_iter=30000,
       momentum=0.9, n_iter_no_change=100, nesterovs_momentum=True,
       power_t=0.5, random_state=5, shuffle=True, solver='adam',
       tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
    
    mlp.fit(X_train1,y_train['h'].values.ravel())

    predictions = mlp.predict(X_test1)
 
    actual = y_test['h'].astype('float64').values.ravel()

    n = actual.shape[0]
    
    print('ANN R2=',r2_score(actual, predictions))
    print('ANN Adj_R2_Score=',(1-(1-r2_score(actual, predictions))*(n-1)/(n-p-1)))
    
    # MAE of test dataset
    results = pd.DataFrame({'Pred':predictions, 'Act':y_test['h'].values.ravel()
                            ,'htp':y_test['htp'].values.ravel(),'hann':y_test['hann'].values.ravel()
                            ,'filename':y_test['filename']})    
    
    results['Diff'] = abs((1-(results['Pred']/results['Act']))*100) 
    print('ANN MAE=', results['Diff'].mean())
    
    # Write results in a dataframe
    ##xlsfile = 'MLP.xlsx'
    ##writer = pd.ExcelWriter(xlsfile, engine='xlsxwriter')
    ##results.to_excel(writer, sheet_name='MLP',startrow=0, startcol=0, header=True ,index=True)  
    
    # Write results in a dataframe
    xlsfile = 'MLP.xlsx'
    results.to_excel(xlsfile,sheet_name ='Sheet1',engine='xlsxwriter')
    
    X_train1a = scaler.transform(X_1a)
    predictions1 = mlp.predict(X_train1a)
    actual1 = df_tar_a['h'].astype('float64').values.ravel()

    results1 = pd.DataFrame({'Pred':predictions1, 'Act':actual1
                             ,'htp':df_tar_a['htp'].values.ravel(),'hann':df_tar_a['hann'].values.ravel()
                             ,'filename':df_tar_a['filename']})

    results1['Diff'] = abs((1-(results1['Pred']/results1['Act']))*100)
    results1.to_excel('MLP_excluded.xlsx',sheet_name = 'Sheet1', engine='xlsxwriter') 
    
    
    
    
    
    
    
    return