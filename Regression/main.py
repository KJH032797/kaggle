import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# print(train.info())
# 매매가 상관계수 상위 4 확인
cor=train.corr(numeric_only=True)['SalePrice'].abs().sort_values(ascending=False)
# print(cor.head())

# 확인한 컬럼 선별 요소 지정
features=['OverallQual','GrLivArea','GarageCars','GarageArea']

# 결측치 처리
# 학습용(정답지)
X_train = train[features].fillna(0)
# 실습용(문제지)
X_test = test[features].fillna(0)
# 답안지
y=train['SalePrice']

# 모델학습(회귀Regression)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y)

# 예측
test['SalePrice']=model.predict(X_test)
test[['Id','SalePrice']].to_csv('data/test_submission.csv',index=False)