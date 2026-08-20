import pandas as pd
import seaborn as sns

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train['Age']=train['Age'].fillna(train.groupby('Pclass')['Age'].transform('median'))
print(train.isna().sum())

