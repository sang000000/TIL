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
#titanic["age"].fillna(titanic["age"].median(), inplace = True)
titanic["age"].fillna( titanic["age"].median(), inplace = True)
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


# 도전과제
#영화 리뷰 감성 분석
# 1. 데이터 셋 불러오기 - 데이터셋을 다운로드한 후에 데이터셋을 불러오고, 불러온 데이터프레임의 상단 5개의 데이터와 하단 5개의 데이터, 컬럼과 shape를 불러오는 코드를 작성해라.
import pandas as pd
df = pd.read_csv("netflix_reviews.csv")
print(df.head())
print(df.tail())
print("Shape of the dataset:",df.shape)
print("Columns in the dataset:",df.columns)
print("score:",df["score"].isnull().sum())
print("score sum:",df["score"].sum())
print("score 5점:",df["score"].value_counts())
#2. 데이터 전처리 - 텍스트 데이터에는 불용어(쓸모없는 단어, 구두점 등)가 많다. 해당 부분을 없애주는 처리가 필요하다. 텍스트 데이터에 대한 전처리를 해라.

# 전처리 함수
import re

text = lambda x: ""if isinstance(x, float) else re.sub(r'\d+', '', re.sub(r'[^\w\s]', '', x.lower())).strip()

for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].apply(text)

print(df.head())

#3. feature 분석 (EDA) - 데이터를 잘 불러오셨다면 해당 데이터의 feature를 찾아야 한다. 해당 넷플릭스의 데이터에는 리뷰가 1점부터 5점까지 있다. 해당 데이터의 분포를 그래프로 그려라. 

import seaborn as sns  # 그래프를 그리기 위한 seaborn 라이브러리 임포트 (없으면 설치 바랍니다)
import matplotlib.pyplot as plt  # 그래프 표시를 위한 pyplot 임포트

plt.figure(figsize= (10,5))
count = df['score'].value_counts().sort_index()
sns.barplot(x = count.index, y = count.values, hue=count.index, palette='Accent', )
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Distribution of Scores')
plt.show()

#4. 리뷰 예측 모델 학습시키기 (LSTM) - 이제 어떤 리뷰를 쓰면 점수가 어떻게 나올지에 대해서 예측을 해보고 싶다. 로지스틱 회귀 등을 사용하여, 리뷰에 대한 점수 예측을 진행해라

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.nn.utils.rnn import pad_sequence

label_encoder = LabelEncoder()
df["score"] = label_encoder.fit_transform(df["score"])

train_reviews, test_reviews, train_ratings, test_ratings = train_test_split(df["content"].values, df['score'].values, test_size=0.2, random_state=42)
# 자연어 처리
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(df["content"]))

# 데이터셋 클래스 정의
class ReviewDataset(Dataset):
    def __init__(self, reviews, ratings, text_pipeline, label_pipeline):
        self.reviews = reviews
        self.ratings = ratings
        self.text_pipeline = text_pipeline
        self.label_pipeline = label_pipeline

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.text_pipeline(self.reviews[idx])
        rating = self.label_pipeline(self.ratings[idx])
        return torch.tensor(review, dtype=torch.long), torch.tensor(rating, dtype=torch.long)
    

def text_pipeline(text):
    tokens = tokenizer(text)
    return vocab(tokens)

def label_pipeline(label):
    return label



# 데이터셋 정의
train_dataset = ReviewDataset(train_reviews, train_ratings, text_pipeline, label_pipeline)
test_dataset = ReviewDataset(test_reviews, test_ratings, text_pipeline, label_pipeline)

# 데이터 로더 정의
BATCH_SIZE = 64

def collate_fn(batch):  # 배치 내 텍스트와 레이블을 패딩합니다.
    texts, labels = zip(*batch)  # 배치에서 텍스트와 레이블을 분리합니다.
    texts_padded = pad_sequence(texts, batch_first=True )  # 텍스트를 패딩합니다.
    labels_tensor = torch.tensor(labels)  # 레이블을 텐서로 변환합니다.
    return texts_padded, labels_tensor

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn = collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn = collate_fn)

# LSTM 모델 정의 
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden[-1])

# 하이퍼파라미터 정의
VOCAB_SIZE = len(vocab)
EMBED_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 5 # 예측할 점수 개수

# 모델 초기화
model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 모델 학습은 직접 작성해보세요!!!


# 모델 학습
num_epochs = 1  # 에포크 수 설정
for epoch in range(num_epochs):
    model.train()  # 모델을 훈련 모드로 설정
    total_loss = 0
    for i, (texts, labels) in enumerate(train_dataloader):  # 훈련 데이터로 반복
        optimizer.zero_grad()  # 기울기 초기화
        outputs = model(texts)  # 모델 출력 계산

        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트
        total_loss += loss.item()
        # 10 Step 마다 손실 출력
        if (i + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}')

    print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss/len(train_dataloader):.4f}')  # 에포크와 손실 출력

        # 매 에포크 마다 모델 평가 실행
    correct = 0
    total = 0
    with torch.no_grad():  # 평가 시에는 기울기 계산을 하지 않음
        for texts, labels in test_dataloader:
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'에포크{epoch+1} - Accuracy: {100 * correct / total}%')



# 예측 함수(예시)
def predict_review(model, review):
    model.eval()
    with torch.no_grad():
        tensor_review = torch.tensor(text_pipeline(review))
        output = model(tensor_review)
        prediction = output.argmax(1).item()
        return label_encoder.inverse_transform([prediction])[0]

# 새로운 리뷰에 대한 예측
new_review = "This app is great but has some bugs."
predicted_score = predict_review(model, new_review)
print(f'Predicted Score: {predicted_score}')