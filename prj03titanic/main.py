import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# df 준비 : 파일 읽어오기
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# df 전처리 : 결측치 , 중복, 인코딩 , 파생칼럼
for df in [train, test]:
    df.dropna(inplace=True)
    df["Sex"] = (df["Sex"] == "female").astype(int)

# df 에서 특징 고르기
X = train[["Sex"]]
y = train["Survived"]

# 학습
m = RandomForestClassifier(n_estimators=100 , random_state=42, max_depth=1)
m.fit(X, y)

# 예측
result = m.predict(test[["Sex"]])
test["Survived"] = result

# 파일로 저장
test.to_csv('data/result.csv', index=False)
