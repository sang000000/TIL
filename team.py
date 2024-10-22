# 1. 데이터셋 불러오기 
import seaborn as sns

titanic = sns.load_dataset('titanic')
print(titanic)

#2. feature 분석
##2-1 head 함수를 이용해 데이터 프레임의 첫 5행을 출력하여 어떤 feature들이 있는지 확인해fk. 
print(titanic.head())

##2-2 describe 함수를 통해서 기본적인 통계를 확인해주세요. 
print(titanic.describe())

##2-3 describe 함수를 통해 확인할 수 있는 count, std, min, 25%, 50%, 70%, max 가 각각 무슨 뜻인지 주석 혹은 markdown 블록으로 간단히 설명해라.
''' count는 컬럼별 총 데이터의 개수를 의미한다.
    std는 컬럼별 데이터의 표준편차을 의미한다.
    min과 max는 컬럼별 데이터의 최대,최소를 의미한다.
    25% 50% 70%는 4분위 수를 기준으로 각가에 해당하는 값들을 의미한다''' 

##2-4 isnull() 함수와 sum()  함수를 이용해 각 열의 결측치 갯수를 확인해라.
print(titanic.isnull().sum())

#3. feature engineering은 모델의 성능을 향상시키기 위해 중요한 단계이다. 2번 feature 분석에서 얻은 데이터에 대한 이해를 바탕으로 아래의 feature engineering을 수행해라.
##3-1 Age(나이)의 결측치는 중앙값으로, Embarked(승선 항구)의 결측치는 최빈값으로 대체해라. 모두 대체한 후에, 대체 결과를 isnull() 함수와 sum()  함수를 이용해서 확인해라.
titanic["age"].fillna(titanic["age"].median(), inplace = True)
titanic["embarked"].fillna(titanic["embarked"].mode()[0], inplace=True)
print(titanic[["age","embarked"]].isnull().sum())

##3-2 Sex(성별)를 남자는 0, 여자는 1로 변환해라. alive(생존여부)를 True는 1, False는 0으로 변환해라. Embarked(승선 항구)는 ‘C’는 0으로, Q는 1으로, ‘S’는 2로 변환해라. 모두 변환한 후에, 변환 결과를 head 함수를 이용해 확인해라. 
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
titanic['alive'] = titanic['alive'].map({'no': 1, 'yes': 0})
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2,})
print(titanic[["sex", "alive", "embarked"]].head())

##3-3 SibSip(타이타닉호에 동승한 자매 및 배우자의 수), Parch(타이타닉호에 동승한 부모 및 자식의 수)를 통해서 family_size(가족크기)를 생성해라. 새로운 feature를 head 함수를 이용해 확인해라. 
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

print(titanic['family_size'].head())

##4-1  학습에 필요한 feature은 'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', ‘family_size’ 입니다. feature과 target을 분리해라.  그 다음 데이터 스케일링을 진행해라. 
titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]
X = titanic.drop('survived', axis=1) # feature
y = titanic['survived'] # target

##4-2 Logistic Regression, Random Forest, XGBoost를 통해서 생존자를 예측하는 모델을 학습하라. 학습이 끝난 뒤 Logistic Regression과 Random Forest는 모델 accuracy를 통해, XGBoost는 mean squared error를 통해 test data를 예측하라.
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

##4-3 Random Forest
from sklearn.tree import DecisionTreeClassifier
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 생성 및 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

##4-4
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost 모델 생성
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 모델 학습
xgb_model.fit(X_train_scaled, y_train)

# 예측
y_pred_xgb = xgb_model.predict(X_test_scaled)

# 평가
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost 모델의 MSE: {mse_xgb}')

