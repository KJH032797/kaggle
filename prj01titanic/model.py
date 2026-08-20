import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

test=pd.read_csv('data/test.csv')
train=pd.read_csv('data/train.csv')
# test['Survived']=(test['Sex']=='female').astype(int)
# test[['PassengerId','Survived']].to_csv('data/result.csv',index=False)

# 전처리
for df in [train,test]:
    df['Age']=df['Age'].fillna(df.groupby('Pclass')['Age'].transform('median'))
    df['Sex']=(df['Sex']=='female').astype(int)
    df['Fare']=df['Fare'].fillna(df['Fare'].median())
    # 특징 생성
    df['FamilySize']=df['SibSp']+df['Parch']+1
    df['Title']=df['Name'].str.extract(r',\s*([^\.]+)\.')
    df['Title']=(df['Title'].replace(['Mlle','Ms'],'Miss')
                 .replace('Mme','Mrs'))
    df['Title']=df['Title'].replace([
        'Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'
    ],
        'Rare')
    df['Title']=df['Title'].map({'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5})

# 특징 선별
features=['Pclass','Sex','Age','Fare','FamilySize','Title']
X=train[features]
y=train['Survived']

# 학습
m=RandomForestClassifier(n_estimators=1000,max_depth=3,random_state=100)
m.fit(X, y)

# 예측
result=m.predict(test[features])
test['Survived']=result
test[['PassengerId','Survived']].to_csv('data/result.csv',index=False)