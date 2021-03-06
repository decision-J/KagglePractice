# baseline을 만든 feature에 lecture_note feature 추가

import pandas as pd
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from keras import models
from keras import layers

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

x_train = train.loc[:, train.columns != 'TripType']
x_test = test.loc[:, test.columns != 'TripType']

train_labels = train.loc[:, train.columns == 'TripType']

# Train label이 순서대로 되어있지 않아 to_categorical 사용하기 위해서 rank값으로 바꿔줌
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
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(38, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label = 'valid loss')
plt.title('T')
plt.xlabel('Epochs')
plt.ylabel('LOSS')
plt.legend()

plt.show()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.clf()
plt.plot(epochs, acc, 'bo', label='Training ')
plt.plot(epochs, val_acc, 'b', label = 'valid ')
plt.title('T')
plt.xlabel('Epochs')
plt.ylabel('ACC')
plt.legend()

plt.show()

### Fit the model
model = KerasClassifier(build_fn=build_model, epochs=10, batch_size=128, verbose=0)
model.fit(x_train, one_hot_train_labels)

### Get prediction
predictions = model.predict_proba(x_test)
predictions
predictions.shape

pd.DataFrame(predictions).to_csv('predictions.csv', index=False)
# Kaggle 기준 score 7.80. 전체 800등 정도
# 성능이 엄청 떨어졌다. 왜일까?
# 변수가 너무 많이 늘어나버려서 64 Dense로는 너무 작은것일까?

# 두번째 시도: Dense 층을 128로 늘림
# Kaggle 기준 score 2.27. 전체 655등 정도
