import pandas as pd
import os
import zipfile

import os


netflix_df = pd.read_csv('netflix_reviews.csv')

print("\n컬럼 이름:\n", netflix_df.shape)

print("\n데이터프레임의 모양 (shape):\n", netflix_df.columns)
 
import re


# 전처리 함수 정의
preprocess_text = lambda text: (
    "" if isinstance(text, float) else
    re.sub(r'\d+', '', re.sub(r'[^\w\s]', '', text.lower())).strip()
)
#람다함수 
#람다 함수는 일반 함수에 비해 문법이 간단합니다. 예를 들어, 일반 함수를 정의할 때는 def 키워드와 함수 이름, 매개변수를 써야 하지만, 람다 함수는 단순히 lambda 키워드 다음에 매개변수와 표현식을 작성하는 방식으로 정의할 수 있습니다. 이로 인해 코드가 짧고 명료해집니다. 작용하기전 결과 
# 전처리 함수
def preprocess_text(text):
    if isinstance(text, float):
        return ""
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\d+', '', text)  
    text = text.strip() 

    return text

preprocess_text = lambda text: (
    "" if isinstance(text, float) else
    re.sub(r'\d+', '', re.sub(r'[^\w\s]', '', text.lower())).strip()
)

 
 
 

for column in netflix_df.select_dtypes(include=['object']).columns:
    netflix_df[column] = netflix_df[column].apply(preprocess_text)
 

# for 문 코드가 하는 역할
#netflix_df에서 문자열 데이터 타입(object)을 포함하는 컬럼을 찾는다:
#for column in netflix_df.select_dtypes(include=['object']).columns:
#각 컬럼에 대해 다음 작업을 반복적으로 수행한다:
#netflix_df[column] = netflix_df[column].apply(preprocess_text)
#(여기서 preprocess_text는 전처리 함수이다.)
#즉 netflix_df에서 문자열 데이터 타입(object)을 포함하는 컬럼 내부의 값을 
#netflix_df에서 문자열 데이터 타입(object)을 포함하는 컬럼 내부의 값을 전처리 함수를 적용시킨다


print(netflix_df.head())

netflix_df.to_csv('processed_file.csv', index=False)

import seaborn as sns  # 그래프를 그리기 위한 seaborn 라이브러리 임포트 (없으면 설치 바랍니다)
import matplotlib.pyplot as plt  # 그래프 표시를 위한 pyplot 임포트

review_counts = netflix_df['score'].value_counts().reset_index()
review_counts.columns = ['score', 'count']

# 바플롯 그리기
palette = ['blue',  'orange', 'green', 'red','purple']  # 각 색깔 지정
sns.barplot(x='score', y='count', data=review_counts, palette=palette)  # 색깔 적용
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Distribution of Scores')
plt.show()



