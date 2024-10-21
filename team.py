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

#3-3 SibSip(타이타닉호에 동승한 자매 및 배우자의 수), Parch(타이타닉호에 동승한 부모 및 자식의 수)를 통해서 family_size(가족크기)를 생성해라. 새로운 feature를 head 함수를 이용해 확인해라. 
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

print(titanic['family_size'].head())



