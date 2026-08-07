import numpy as np
import pandas as pd #allows us to read and edit the dataset
import sklearn.linear_model
from matplotlib import pyplot as plt, pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import linear_model, __all__
from sklearn import svm
from xgboost import XGBClassifier
import seaborn as sns
from Data_Cleaning import *

#wednesday refactoring model
# thursday834-11, research, expermienting with traing and testing size, added xgboost
#friday 9-940, 11-11:35
#sat 2-4 
def cleandata():
    data = pd.read_excel("V4_Cleaned_TRANSLATED_2020_Testing_Group.xlsx")
    looking_for = 'Had aGVHD (aGVHD)'
    df = label_encode(set_none(data), looking_for, False,drop_GVHD())
    # NOTE: we need to import random forest
    # Insert cleaning code ex: drop education or n/a rows
    # Drop NA
    X = df.drop(columns=[looking_for])  # this is our input set (things we need to check to find our output) TESTING
    feature_names = [f" {col}" for col in X.columns]
    Y = df[looking_for]  # output data set what we ant to get TRAINING


    #X_train = X
    #X_test = X
    #Y_train = Y
    #Y_test = Y




    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)  # sets aside 20% of data for testing

    print(Y_train)
    print(X_train)




    X_test = X
    Y_test = Y

    # todo: 80 training, all testing
    # first 2 are input training + testing second 2 are forout teting + trainn
    # scales the numbers so things have an equal impact
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    return X_train,X_test,Y_train,Y_test,feature_names


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

def randforest(MAXLeafnodes,X_train,X_test,Y_train,Y_test,feature_names) -> None:

    rfModel = RandomForestClassifier(max_leaf_nodes=3,max_depth=6,n_estimators=50,max_features=None)
    rfModel.fit(X_train,Y_train)
    rfPred = rfModel.predict(X_test)
    Score_RF = accuracy_score(Y_test, rfPred)
    mae = mean_absolute_error(Y_test, rfPred)
    print("Randforest mae and accuracy")
    print(mae , "" , Score_RF)
    print(Score_RF)
    print(classification_report(Y_test, rfPred))
    print(confusion_matrix(Y_test, rfPred))

    global_importances = pd.Series(rfModel.feature_importances_, index=feature_names)
    global_importances.sort_values(ascending=False, inplace=True)
    global_importances.head(10).plot.barh(color='green')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Global Feature Importance - Built-in Method")
    plt.show()


def RANmaxleaftest() -> None: #Checks if overfit or undefit for parametets
    X_train, X_test, Y_train, Y_test = cleandata()
    for max_leaf_nodes in [5, 50, 100, 200, 500]: ##50 or 200 for parameter optimization
        myranfor = randforest(max_leaf_nodes,X_train, X_test, Y_train, Y_test)
def svmodel(X_train,X_test,Y_train,Y_test,feature_names)->None:
    svcmodel = SVC(kernel='rbf', C=2)
    svcmodel.fit(X_train,Y_train)
    perm_importance = permutation_importance(svcmodel, X_test, Y_test)
    #TODO: FIX :(
    features = np.array(feature_names)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.show()
    svcpred = svcmodel.predict(X_test)
    svc_score = accuracy_score(Y_test, svcpred)
    mae = mean_absolute_error(Y_test,svcpred)
    print(mae , "" , svc_score)


def SGD(X_train,X_test,Y_train,Y_test,feature_names)-> None: #Better error wise
    sgdcl = linear_model.SGDClassifier(max_iter = 1000, tol=1e-3,penalty = "elasticnet")
    sgdcl.fit(X_train,Y_train)
    pred = sgdcl.predict(X_test)

    Score_RF = accuracy_score(Y_test, pred)
    mae = mean_absolute_error(Y_test, pred)
    print("SGD mae and accuracy")
    print(mae, "", Score_RF)
    print(Score_RF)
    print(classification_report(Y_test, pred))
    print(confusion_matrix(Y_test, pred))


def XGBoost(X_train,X_test,Y_train,Y_test,feature_names):
    bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    bst.fit(X_train, Y_train)
    pred = bst.predict(X_test)
    Score_RF = accuracy_score(Y_test, pred)
    mae = mean_absolute_error(Y_test, pred)
    print("XGBoost mae and accuracy")
    print(mae, "", Score_RF)
    print(Score_RF)
    print(classification_report(Y_test, pred))
    print(confusion_matrix(Y_test, pred))
    global_importances = pd.Series(bst.feature_importances_, index=feature_names)
    global_importances.sort_values(ascending=True, inplace=True)
    global_importances.plot.barh(color='green')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Global Feature Importance - Built-in Method")
    plt.show()

#TODO: summary report
class LRE():
    def __init__(self,name,range):
        self.name = name
        self.range = range

    def getOutput(self):
        print(self.name, (self.range))
def linerRegress(X_train,X_test,Y_train,Y_test,feature_names)-> None:




    LR = linear_model.LinearRegression()
    LR.fit(X_train,Y_train)


    LRPred = LR.predict(X_test)
    mae = mean_absolute_error(Y_test, LRPred)
    print("Linear Regression mae")
    print(mae)

    importance = LR.coef_


    featImport = []
    for i, v in enumerate(importance):
        importantfeaturesitems = LRE(feature_names[i], ', Score: %.5f' % (v))
        featImport.append(importantfeaturesitems)


    n = 10
    new_list = sorted(featImport, key=lambda x: x.range, reverse=True)

    for i in range(n):
        print(new_list[i].getOutput())

    print("")
    print("")
    print("")

    revnewlist = new_list[::-1]
    for i in range(n):
        print(revnewlist[i].getOutput())



X_train, X_test, Y_train, Y_test, feature_names = cleandata()
randforest(200,X_train, X_test, Y_train, Y_test, feature_names)
linerRegress(X_train, X_test, Y_train, Y_test, feature_names)
#XGBoost(X_train, X_test, Y_train, Y_test, feature_names)
