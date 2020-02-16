# RF model try

import pandas as pd
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import os
os.getcwd()
os.chdir('C:/Users/JYW/OneDrive - 연세대학교 (Yonsei University)/유용코드/2020 Learning from TOP Kegglers/Recommeded Competitions/walmart')

train = pd.read_csv("C:/Users/JYW/OneDrive - 연세대학교 (Yonsei University)/유용코드/2020 Learning from TOP Kegglers/Recommeded Competitions/walmart/train.csv")
test = pd.read_csv("C:/Users/JYW/OneDrive - 연세대학교 (Yonsei University)/유용코드/2020 Learning from TOP Kegglers/Recommeded Competitions/walmart/test.csv")
sample_sub = pd.read_csv("C:/Users/JYW/OneDrive - 연세대학교 (Yonsei University)/유용코드/2020 Learning from TOP Kegglers/Recommeded Competitions/walmart/sample_submission.csv")


# 데이터 전처리
## 1. Upc 제거: FinelineNumber와 동일 정보
del train["Upc"]
del test["Upc"]

## 2. DepartmentDescription와 FinelineNumber의 NA제거
## test set은 mode로 대체
train.isnull().sum()
train = train.dropna()

test.isnull().sum()
train[["VisitNumber",'FinelineNumber']].groupby("FinelineNumber").count().idxmax()
test["FinelineNumber"] = test["FinelineNumber"].fillna(5501.0)
train[["VisitNumber",'DepartmentDescription']].groupby("DepartmentDescription").count().idxmax()
test["DepartmentDescription"] = test["DepartmentDescription"].fillna("GROCERY DRY GOODS")

## 3. 범주형 변수 처리
### 3-1. 요일은 주중, 주말로.
weekend = ["Saturday","Sunday","Friday"]
train['Week'] = list([1 if train['Weekday'].iloc[i] in weekend else 0 for i in range(0, len(train))])
del train['Weekday']

test['Week'] = list([1 if test['Weekday'].iloc[i] in weekend else 0 for i in range(0, len(test))])
del test['Weekday']

## 4. VisitNumber가 Key로 다양한 품목 데이터들이 중복되어 있는 데이터
## Key를 기준으로 groupby를 해서 모든 정보를 반영해주어야 TripType을 예측하는데 정확한 정보일 것이라 판단

### 4-1. ScanCount와 Week 변수는 동일한 값. 따라서 맨 윗 값만 반영하면 됨
### 4-2. FinelineNumber: 항목별 Count가 0~2000이상 까지 다양함
###      Baseline 이므로 일단 Count 값으로 변환하여 VisitNumber 별로 평균치를 반영해보도록 함: FineLineCount
###     이유는 사람들이 많이 구입하는 품목을 사는 타입, 마이너한 품목을 사는 타입으로 반영되도록 한 것임
###     또한, FinelineNumber를 몇개 갖고있는지(=몇 개의 품목을 샀는지) 정보도 반영: FineLineSum
FineLineSum = train[['VisitNumber', 'FinelineNumber']].groupby('VisitNumber').count()
train = pd.merge(train, FineLineSum, on=["VisitNumber"], how="left")
train.rename(columns={'FinelineNumber_x':'FinelineNumber','FinelineNumber_y':'FineLineSum'}, inplace=True)

FineLineCount = train[['VisitNumber', 'FinelineNumber']].groupby('FinelineNumber').count()
train = pd.merge(train, FineLineCount, on=["FinelineNumber"], how="left")
train.rename(columns={'VisitNumber_x':'VisitNumber','VisitNumber_y':'FineLineCount'}, inplace=True)

train = pd.merge(train, train[['VisitNumber', 'FineLineCount']].groupby('VisitNumber').mean(), on=["VisitNumber"], how="left")
train.rename(columns={'FineLineCount_y':'FineLineCount'}, inplace=True)
train

del train['FineLineCount_x']
del train['FinelineNumber']

train = train.drop_duplicates(["VisitNumber"])

FineLineSum = test[['VisitNumber', 'FinelineNumber']].groupby('VisitNumber').count()
test = pd.merge(test, FineLineSum, on=["VisitNumber"], how="left")
test.rename(columns={'FinelineNumber_x':'FinelineNumber','FinelineNumber_y':'FineLineSum'}, inplace=True)

FineLineCount = test[['VisitNumber', 'FinelineNumber']].groupby('FinelineNumber').count()
test = pd.merge(test, FineLineCount, on=["FinelineNumber"], how="left")
test.rename(columns={'VisitNumber_x':'VisitNumber','VisitNumber_y':'FineLineCount'}, inplace=True)

test = pd.merge(test, test[['VisitNumber', 'FineLineCount']].groupby('VisitNumber').mean(), on=["VisitNumber"], how="left")
test.rename(columns={'FineLineCount_y':'FineLineCount'}, inplace=True)
test

del test['FineLineCount_x']
del test['FinelineNumber']

test = test.drop_duplicates(["VisitNumber"])

### (NEW!!) 3-2. DepartmentDescription을 범주화시켜 반영(lecture_note 내용)
gb = train.groupby('VisitNumber')["DepartmentDescription"].value_counts().unstack().fillna(0).reset_index()
train = train.reset_index()
del train["index"]

train = pd.concat([train, gb], axis=1)

gb_test = test.groupby('VisitNumber')["DepartmentDescription"].value_counts().unstack().fillna(0).reset_index()
list(set(list(gb.columns)) - set(list(gb_test.columns)))
# gb_test에는 없는 컬럼이 존재한다. 만들어주어야 shape이 맞게됨
gb_test["HEALTH AND BEAUTY AIDS"] = 0.0

test = test.reset_index()
del test["index"]

test = pd.concat([test, gb_test], axis=1)

del train["DepartmentDescription"]
del test["DepartmentDescription"]

# Model Builing
## 1. 데이터 나누기 (X, y)
del train["VisitNumber"]
del test["VisitNumber"]

from sklearn.model_selection import cross_val_score, train_test_split

train_val, valid = train_test_split(train, test_size=0.2, random_state=0)

x_train = train_val.loc[:, train_val.columns != 'TripType']
x_valid = valid.loc[:, valid.columns != 'TripType']
x_test = test.loc[:, test.columns != 'TripType']

train_labels = train_val.loc[:, train_val.columns == 'TripType']
val_labels = valid.loc[:, valid.columns == 'TripType']

x = train.loc[:, train.columns != 'TripType']
labels = train.loc[:, train.columns == 'TripType']

## 2. 데이터 정규화 for DL
# mean = x_train.mean(axis=0)
# x_train -= mean
# std = x_train.std(axis=0)
# x_train /= std
#
# x_test -= mean
# x_test /= std
## test data를 정규화 시킬 때도 train data의 mean과 std를 사용해서 한다!
### RF는 정규화가 사실상 의미없음


## 3. modeling
### 3-1. Base model check
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(x_train, train_labels)

### Get prediction
predictions = clf.predict_proba(x_test)
predictions
predictions.shape

pd.DataFrame(predictions).to_csv('predictions_base_RF.csv', index=False)
# Kaggle 기준 score 2.26.
# 그냥 해본 base RF가 딥러닝만큼 좋네..
# 반도체 분석 때도 RF가 성능이 뛰어났던 case 존재


### 3-2. Hyperparameter tuning
from bayes_opt import BayesianOptimization

n_folds = 3
random_seed=6

# BayesianOptimization
def eval(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    params["n_estimators"] = max(n_estimators, 1)
    params['max_depth'] = max(max_depth, 1)
    params['min_samples_split'] = max(min_samples_split, 0)
    params['min_samples_leaf'] = max(min_samples_leaf, 0)
    rfc = RandomForestClassifier()
    cv_result = cross_val_score(rfc, x_train, train_labels, cv=3, scoring='accuracy')

    return min(cv_result)

params = {'max_depth': (10, 500)
          ,'min_samples_leaf': (1, 10)
          ,'min_samples_split': (1, 10)
          ,'n_estimators': (10, 500)
          # ,'bootstrap': [True, False]
          # ,'max_features': ['auto', 'sqrt']
          }


rfc_optimization = BayesianOptimization(eval, params, random_state=0)
init_round = 5
opt_round = 15

rfc_optimization.maximize(init_points=init_round, n_iter=opt_round)

rfc_optimization.max
params = rfc_optimization.max["params"]

op_clf = RandomForestClassifier(n_estimators=int(max(params["n_estimators"], 0)),
                                max_depth=int(max(params["max_depth"], 1)),
                                min_samples_split=int(max(params["min_samples_split"], 2)),
                                min_samples_leaf=int(max(params["min_samples_leaf"],2)), random_state=0)
op_clf.fit(x, labels)

### Get prediction
predictions = op_clf.predict_proba(x_test)
predictions
predictions.shape

pd.DataFrame(predictions).to_csv('predictions_tuning_RF.csv', index=False)
# Kaggle 기준 score 2.09.
# 조금 더 좋아졌다.
