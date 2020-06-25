import numpy as np
import pandas as pd
import matplotlib as plt
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split

numerical = ["Pricing, Delivery_Terms_Quote_Appr", "Pricing, Delivery_Terms_Approved", 
             "Bureaucratic_Code_0_Approval", "Bureaucratic_Code_0_Approved", "Submitted_for_Approval", 
             "Opportunity_ID", "ASP", "ASP_(converted)", "Delivery_Year", "TRF", "Total_Amount", 
             "Total_Taxable_Amount", "Planned_Delivery_End_Date", "Planned_Delivery_Start_Date", 
             "Account_Created_Date", "Opportunity_Created_Date", "Quote_Expiry_Date", "Last_Modified_Date"]

useless = ["ID","Last_Activity", "ASP_converted_Currency", "Prod_Category_A","Actual_Delivery_Date", "Submitted_for_Approval", "Opportunity_Name", "Account_Type", "Delivery_Terms", "Size", "Price", "ASP_Currency", "Total_Amount_Currency", "Total_Taxable_Amount_Currency"]
dates = ["Account_Created_Date", "Opportunity_Created_Date", "Quote_Expiry_Date", "Last_Modified_Date", "Planned_Delivery_Start_Date", "Planned_Delivery_End_Date"]
target = ["Stage", "Opportunity_ID"]
leak = ["Opportunity_ID"]

def preprocess(data):
    
    # elimino caracteres prohibidos
    
    data = data.rename(columns = lambda x:re.sub("[^A-Za-z0-9_]+", "", x))
    
    # casteo a categoricas varias columnas

    categorical = [x for x in data.columns if x not in numerical and x != target[0]]

    for c in categorical:
        data[c] = data[c].astype('category')

    # limpio columnas
    
    data = data.drop(useless + dates, axis = 1)  

    # agrego feature Contacts: la cantidad de negociaciones registradas

    data["Contacts"] = data.groupby("Opportunity_ID", sort=False)["Opportunity_ID"].transform('count')

    return (data)

def train(file):

    # leo y preproceso los datos

    data = pd.read_csv(file)
    data = preprocess(data)

    # me quedo con las que tienen Stage definido

    data = data[(data.Stage == "Closed Won") | (data.Stage == "Closed Lost")]
    data.Stage = data.Stage.replace({"Closed Won": 1, "Closed Lost": 0})
    
    # armo train y test

    x_train, x_test, y_train, y_test = train_test_split(data.drop(target, axis = 1), data.Stage, test_size = 0.3, random_state = 0)

    categorical = [x for x in data.columns if x not in numerical and x != target[0]]

    train_data = lgb.Dataset(x_train, label = y_train, categorical_feature = categorical)
    test_data = lgb.Dataset(x_test, label = y_test)

    # entreno con estos parametros un lightgbm

    parameters = {
        'application': 'binary',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'is_unbalance': 'true',
        'boosting': 'gbdt',
        'num_leaves': 31,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'learning_rate': 0.05,
        'verbose': 0
    }

    model = lgb.train(parameters, train_data, valid_sets = test_data, num_boost_round = 5000, early_stopping_rounds = 100)

    return (model)

def predict(file_train, file_validation, file_submit):

    # leo y proceso la validacion

    validation = pd.read_csv(file_validation)
    validation = preprocess(validation)

    # entreno el modelo y predigo

    model = train(file_train)

    pred = model.predict(validation.drop(leak, axis = 1))

    # agrupo por Opportunity_ID para dar una sola prediccion por solicitud

    validation_index = validation.index
    pred = pd.DataFrame(pred, index = validation_index, columns = ["Prediction"])
    validation = validation.join(pred)

    answer = validation.groupby("Opportunity_ID")["Prediction"].mean()
    
    # Escribo la respuesta en submit

    answer.to_csv(file_submit, header = False)

    return (answer)

# Uso:
predict("Entrenamieto_ECI_2020.csv", "Validacion_ECI_2020.csv", "submission.csv")