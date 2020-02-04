import pandas as pd

train = pd.read_csv("C:/Users/JYW/OneDrive - 연세대학교 (Yonsei University)/유용코드/2020 Learning from TOP Kegglers/Recommeded Competitions/walmart/train.csv")
test = pd.read_csv("C:/Users/JYW/OneDrive - 연세대학교 (Yonsei University)/유용코드/2020 Learning from TOP Kegglers/Recommeded Competitions/walmart/test.csv")
sample_sub = pd.read_csv("C:/Users/JYW/OneDrive - 연세대학교 (Yonsei University)/유용코드/2020 Learning from TOP Kegglers/Recommeded Competitions/walmart/sample_submission.csv")

train.head()
test.head()
sample_sub.head()
