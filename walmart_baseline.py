# 최대한 간단한 Baseline model을 만들고 성능을 살펴보자
# Upc 제거, Null 제거, Simple MLP classifier

import pandas as pd
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier

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
train[["VisitNumber",'FinelineNumber']].groupby("FinelineNumber").count()
test = test.fillna(0.0)

## 3. 범주형 변수 처리
### 3-1. 요일은 주중, 주말로.
weekend = ["Saturday","Sunday","Friday"]
train['Week'] = list([1 if train['Weekday'].iloc[i] in weekend else 0 for i in range(0, len(train))])
del train['Weekday']

test['Week'] = list([1 if test['Weekday'].iloc[i] in weekend else 0 for i in range(0, len(test))])
del test['Weekday']

### 3-2. DepartmentDescription은 일단 삭제해보자
train["DepartmentDescription"].value_counts()

del train["DepartmentDescription"]
del test["DepartmentDescription"]

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

train.head()
test.head()

# Model Builing
## 1. 데이터 나누기 (X, y)
del train["VisitNumber"]
del test["VisitNumber"]

x_train = train.loc[:, train.columns != 'TripType']
x_test = test.loc[:, test.columns != 'TripType']

train_labels = train.loc[:, train.columns == 'TripType']
rank = pd.DataFrame()
rank["TripType"] = train["TripType"].unique()
rank["rank"] = rank.iloc[:].rank(ascending=True)

train_labels = pd.merge(train_labels, rank, on=["TripType"], how="left")
del train_labels["TripType"]
train_labels = train_labels - 1

one_hot_train_labels = to_categorical(train_labels, num_classes=38)
one_hot_train_labels.shape

## 2. 데이터 정규화 for DL
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std

x_test -= mean
x_test /= std
## test data를 정규화 시킬 때도 train data의 mean과 std를 사용해서 한다!


## 3. modeling
### def model
from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(38, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

### K-fold
k = 3

num_val_samples = len(x_train) // k
num_epochs = 200
all_acc_histories = []
for i in range(k):
    print('처리중인 폴드 #', i)
    # 검증 데이터 준비: k번째 분할
    val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = one_hot_train_labels[i * num_val_samples: (i + 1) * num_val_samples]

    # 훈련 데이터 준비: 다른 분할 전체
    partial_x_train = np.concatenate(
        [x_train[:i * num_val_samples],
         x_train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [one_hot_train_labels[:i * num_val_samples],
         one_hot_train_labels[(i + 1) * num_val_samples:]],
        axis=0)

    # 케라스 모델 구성(컴파일 포함)
    model = build_model()

    history = model.fit(partial_x_train, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=128, verbose=0)
    acc_history = history.history['val_accuracy']
    all_acc_histories.append(acc_history)
    print(i, ' 폴드 끝남')

### Score check
average_acc_history = [
    np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_acc_history) + 1), average_acc_history)
plt.xlabel('Epochs')
plt.ylabel('Validation ACC')
plt.show()

### Fit the model
model = KerasClassifier(build_fn=build_model, nb_epoch=100, batch_size=128, verbose=0)
model.fit(x_train, one_hot_train_labels)

### Get prediction
predictions = model.predict_proba(x_test)
predictions
predictions.shape

pd.DataFrame(predictions).to_csv('predictions.csv', index=False)
# Kaggle 기준 score 2.36. 전체 600등 정도
