from datetime import datetime

import numpy as np
import pandas
import pandas as pd
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import  MinMaxScaler

warnings.simplefilter("ignore")
# standard scaler
# MULTIPLE IMPUTATION
# https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-017-0442-1
# https://www.geeksforgeeks.org/imputing-missing-values-before-building-an-estimator-in-scikit-learn/

# TODO: one hot encoding for dataset
# TODO: replace / with dataset

# https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/
# https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/

# Data with slashes means that it doesn't apply to the person, do not put data if there is a row with a slash
#################
# IF UNUSED DELETE ELSE IGNORE
##
data = pd.read_excel("V4_Cleaned_TRANSLATED_2020_Testing_Group.xlsx")

GVHD_Patients = pd.DataFrame()
GVHD_Set_None = pd.DataFrame()
No_GVHD_Patients = pd.DataFrame()
Alive_Patients = pd.DataFrame()
Patients_with_recurrnces = pd.DataFrame()
Dropped_Exclusive_GVHD_items = pd.DataFrame()

# disccused concerns

# TODO: have functions all return data
# TODO: create a function that soly prints/exports data (Use function names to export)
# TODO: standardize data, some of the data is in K/ml whole others are per 106kh
# todo: IMPLEMENT ONE HOT ENCONDING
# study some deta processing, attention base
# spark data engineering

# testing in parallel
# use deep learning to find influfence factors
# make preictions of factors
# find corellations of factors and deaths
# change parameters of first layer + last layer, correct dimenisions of matriz, depends on regression vs classification,
# normally use cross entropy for classigication, mse for regression (loss functions)


'''
Things we can do for model:
Use GVHD patient list to attempt to find the cause for it
'''


###################################
# FUNCTIONS FOR RAW DATA CLEANING #
###################################

def baseDroppedTables(data):
    data = data.drop(index=0)
    data = data.drop(columns="Name/Numbers")
    data = data.drop(columns="Case ID")

    # upper case
    data = data.applymap(lambda s: s.lower() if type(s) == str else s)

    data['diagnosis date'] = pd.to_datetime(data['diagnosis date']).dt.date
    data['Transplantation Date'] = pd.to_datetime(data['Transplantation Date']).dt.date
    data['diagnosis date'] = data['diagnosis date'].apply((pd.Timestamp))
    data['Transplantation Date'] = data['Transplantation Date'].apply((pd.Timestamp))
    data['Days between Lukemia Diagnosis and Stem Transplant'] = data['Transplantation Date'] - data['diagnosis date']
    data['Days between Lukemia Diagnosis and Stem Transplant'] = (
    data['Days between Lukemia Diagnosis and Stem Transplant']).dt.days
    data = data.drop(columns='diagnosis date')
    data = data.drop(columns='Transplantation Date')

    # converts floats that don't need to be floats into ints
    data['Patient Age'] = data['Patient Age'].astype(int)
    data['Number of pre-transplant chemotherapy treatments'] = data[
        'Number of pre-transplant chemotherapy treatments'].astype(int)
    data['degree of compatibility'] = data['degree of compatibility'].astype(int)
    data['Medium granulocyte reconstitution time'] = data['Medium granulocyte reconstitution time'].astype(int)
    data['Platelet reconstitution time'] = data['Platelet reconstitution time'].astype(int)
    data['amount of follow ups'] = data['amount of follow ups'].astype(int)

    # NEW GVHD RULES HERE:
    data['Recurrence amount(relpase / recur (of a disease) )'] = data[
        'Recurrence amount(relpase / recur (of a disease) )'].astype(int)
    data['amount of occurences (cGVHD)'] = data['amount of occurences (cGVHD)'].astype(int)

    # ask AML classification and how to classify it,

    return data


def set_none(data):
    GVHD_Set_None = baseDroppedTables(data).copy()

    GVHD_Set_None['nucleated cells（108/kg）'].replace('/', 0, inplace=True)
    GVHD_Set_None['area of affliciton (aGVHD)'].replace('/', 'none', inplace=True)
    GVHD_Set_None['degree of aGVHD'].replace('/', 'none', inplace=True)
    GVHD_Set_None['strain (infections)'].replace('/', 'none', inplace=True)
    GVHD_Set_None['CMV (infections)'].replace('/', 'none', inplace=True)
    GVHD_Set_None['region of afflection (cGVHD)'].replace('/', 'none', inplace=True)
    GVHD_Set_None['degree of afflecition (cGVHD)'].replace('/', 'none', inplace=True)
    GVHD_Set_None['cGVHD man-made or naturally occurring'].replace('/', 'none', inplace=True)

    return GVHD_Set_None


def label_encode(data, myDroppedItem):
    label_endco = data.copy()
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')

    # Items that need to be label_Encoded
    to_be_encoded = ['Patient Gender', 'Transplantation method', 'AML Classifcation', 'Intial Diagnosis Risk levl',
                     'Mutations/fusion genes', 'Gender of Providor', 'Provider-recipient Gender Relationships',
                     'pre-transplant status', 'Provider-recipient Gender Relationships', 'Pretreatment program',
                     'area of affliciton (aGVHD)', 'degree of aGVHD', 'area of infection (infections)',
                     'CMV (infections)', 'has cGVHD', 'region of afflection (cGVHD)',
                     'cGVHD man-made or naturally occurring', 'case of survival', 'Had aGVHD (aGVHD)',
                     'hematology (relpase / recur (of a disease) )', 'strain (infections)',
                     'degree of afflecition (cGVHD)']

    to_be_encoded = list(set(to_be_encoded))

    # removes items to be encoded for mapping purposes
    to_be_encoded.remove(myDroppedItem)

    # converts labeled item to one-hot into a seperate dataframe
    ohetransform = ohe.fit_transform(label_endco[to_be_encoded])
    # deletes one-hot items from prev datagam
    for i in to_be_encoded:
        label_endco = label_endco.drop(i, axis=1)

    label_endco = label_endco.join(ohetransform)

    # iterates items not dropped to apply one-hot encoding





    # TODO: map mydroppeditem
    uniqueitems = set(label_endco[myDroppedItem])
    uniqueitems = list(uniqueitems)

    for i in range(len(uniqueitems)):
        label_endco[myDroppedItem].replace(uniqueitems[i], i, inplace=True)


    scaler = MinMaxScaler()
    new_df = pd.DataFrame(scaler.fit_transform(label_endco), columns=label_endco.columns, index=label_endco.index)
    #print(label_endco.to_string())

    return new_df


###############################################
# POST DATA CLEANING FUNCTIONS (USE FOR MODEL #
###############################################

def drop_GVHD(data):
    # TODO: adjust how function determines which indexes need to be dropped
    gvhd_Index_start = data.columns.get_loc('area of affliciton (aGVHD)')
    gvhd_Index_end = data.columns.get_loc('degree of aGVHD')
    data.drop(data.iloc[:, gvhd_Index_start:gvhd_Index_end + 1], inplace=True, axis=1)

    gvhd_Index_start_2 = data.columns.get_loc('area of infection (infections)')
    gvhd_Index_end_2 = data.columns.get_loc('degree of afflecition (cGVHD)')
    data.drop(data.iloc[:, gvhd_Index_start_2:gvhd_Index_end_2 + 1], inplace=True, axis=1)

    data = data.drop(columns="cGVHD man-made or naturally occurring")
    data = data.drop(columns="nucleated cells（108/kg）")
    return data


def GVHD(data):
    # TODO: adjust how function determines which indexes need to be dropped
    gvhd_Index_start = data.columns.get_loc('area of affliciton (aGVHD)')
    gvhd_Index_end = data.columns.get_loc('degree of aGVHD')
    gvhd_Index_start_2 = data.columns.get_loc('area of infection (infections)')
    gvhd_Index_end_2 = data.columns.get_loc('cGVHD man-made or naturally occurring')
    print(gvhd_Index_start, gvhd_Index_end, gvhd_Index_start_2, gvhd_Index_end_2)


def Lukemia_Information_Only(data):
    # TODO: adjust how function determines which indexes need to be dropped
    lukemia_only = baseDroppedTables(data).copy()

    # Information that cuts off after Platelet reconsitition time, (ie: before chart with subgroups)
    lukemia_only = lukemia_only.copy()  # Use copy to create, don't create a shallow copy >:(
    lukemia_only.drop(lukemia_only.iloc[:, 19:], inplace=True, axis=1)
    return (lukemia_only)


def Living_Patients_Only(data):
    Alive_Patients = baseDroppedTables(data).copy()
    dead_patients_index = Alive_Patients[Alive_Patients['case of survival'] == 'D'].index
    Alive_Patients.drop(dead_patients_index, inplace=True)
    # print(Alive_Patients.to_string())
    return (Alive_Patients)


def Had_GVHD(data):
    GVHD_Patients = baseDroppedTables(data).copy()
    GVHD_Patients_index = GVHD_Patients[GVHD_Patients['Had aGVHD (aGVHD)'] == 'no'].index
    GVHD_Patients.drop(GVHD_Patients_index, inplace=True)
    return (GVHD_Patients)


def Did_not_Have_GVHD(data):
    No_GVHD_Patients = baseDroppedTables(data).copy()
    GVHD_Patients_index = No_GVHD_Patients[No_GVHD_Patients['Had aGVHD (aGVHD)'] == 'Yes'].index
    No_GVHD_Patients.drop(GVHD_Patients_index, inplace=True)
    return (No_GVHD_Patients)


def Patients_with_recurrences_Only(data):
    Patients_with_recurrnces = baseDroppedTables(data).copy()
    no_hematology_recurrences_index = Patients_with_recurrnces[
        Patients_with_recurrnces['hematology (relpase / recur (of a disease) )'] == 'CR fu '].index
    Patients_with_recurrnces.drop(no_hematology_recurrences_index, inplace=True)
    return (Patients_with_recurrnces.to_string)


#####################
# PROCESS DATA HERE #
#####################
'''
Note: for intial cleaning of rough items use:
baseDroppedTables() first then,
set_none()
'''


