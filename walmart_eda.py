import pandas as pd
import plotnine as p9
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("C:/Users/JYW/OneDrive - 연세대학교 (Yonsei University)/유용코드/2020 Learning from TOP Kegglers/Recommeded Competitions/walmart/train.csv")
test = pd.read_csv("C:/Users/JYW/OneDrive - 연세대학교 (Yonsei University)/유용코드/2020 Learning from TOP Kegglers/Recommeded Competitions/walmart/test.csv")
sample_sub = pd.read_csv("C:/Users/JYW/OneDrive - 연세대학교 (Yonsei University)/유용코드/2020 Learning from TOP Kegglers/Recommeded Competitions/walmart/sample_submission.csv")

train.tail()
test.head()
sample_sub.head()

train.shape
test.shape
# 거의 비슷한 숫자
sample_sub.shape
# test set에 있는 실제 인원 수는 95천명
train.isnull().sum()
test.isnull().sum()
# Train과 Test 모두 Null 존재

########## EDA
#### TripType
train["TripType"] = train["TripType"].apply(lambda x: -1 if x==999 else x)
# 999코드를 -1로 변경
p9.ggplot(train, p9.aes(x='TripType',fill='TripType')) +p9.geom_bar()
# 39와 40이 상당히 많음. 특히 40이 엄청 많음.
train["TripType"].value_counts()
train.info()

#### Weekday
p9.ggplot(train, p9.aes(x='Weekday',fill='Weekday')) +p9.geom_bar()
# 역시 토일이 쇼핑이 많음

#### Upc
len(train['Upc'].unique())
# Upc Number에는 중복값이 많군.
train[train['Upc'].isnull()]
# Upc가 Missing이면 DepartmentDescription, FinelineNumber 모두 Missing.
# 특히, FinelineNumber의 Missing은 Upc와 동일. 두 변수가 밀접한 관계가 있음을 알 수 있음.

train[(train['Upc'].isnull()) & (train['DepartmentDescription'].notnull())]['DepartmentDescription'].value_counts()
# Upc가 Missing인데 DepartmentDescription이 Missing이 아닌 경우는 PHARMACY RX 하나.

word_list = list(train["DepartmentDescription"].dropna())
pd.Series([word for word in word_list if "PHARMACY" in word]).value_counts()
# PHARMACY는 RX외에도 다른 품목이 존재. RX는 매우 소수 품목.

train[(train['Upc'].notnull()) & (train['DepartmentDescription']=="PHARMACY RX")]
# 또한 PHARMACY RX 중에서도 Upc가 있는 애들 존재. (매우 소수)
# 추가적으로 같은 DepartmentDescription에서도 다른 Upc 가지는 case 있음.



#### ScanCount
p9.ggplot(train, p9.aes(x='ScanCount',fill='ScanCount')) +p9.geom_bar() +p9.scale_x_continuous(limits = [-2,7])
# 1이 압도적으로 많음. 최대값은 5. 반품되는 경우는 3위.
train[train['ScanCount']==-1]["DepartmentDescription"].value_counts().head(10)
# 반품되는 가장 많은 품목은 Financial Services. 또한 WEAR도 반품이 많이 됨.

#### FinelineNumber
p9.ggplot(train, p9.aes(x='FinelineNumber',fill='FinelineNumber')) +p9.geom_bar()



#### EDA 후 알게된 주요 포인트:
#### 1.여러개의 행이 동일한 사람의 정보. 즉, 같은 Number의 여러 행 정보를 반영하여 모델을 building하여야 함.
#### 2. TripType 40이 엄청 크기 때문에 저 곳으로 예측확률이 쏠릴 가능성이 다분. 이를 방지하여 모델 학습하여야 함.
