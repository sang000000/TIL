def solution(clothes):
    answer = 0 # 조합의 수를 저장할 변수 초기화
    categorized_items = {} # 의상을 종류별로 분류하기 위한 빈 딕셔너리
    
    # 의상을 종류별 분류
    for item, category in clothes:
        if category not in categorized_items:
            categorized_items[category] = []  # 카테고리가 없으면 새로 생성
        categorized_items[category].append(item) # 해당 카테고리에 의상 추가
    
    a = list(categorized_items.items()) # 각 카테고리의 의상 개수 리스트로 변환
    answer = len(a[0][1])+1 # 첫 번째 카테고리의 경우의 수로 초기화 (의상 선택 + 선택 안 함)
    
    # 모든 카테고리의 조합 수 계산
    for i in range(1,len(a)):
        answer *= len(a[i][1])+1 # 각 카테고리의 의상 개수에 선택 안 함 경우 추가
        
    return answer-1 # 아무것도 안 착용한 경우 제외




#1) 문제 설명

'''코니는 매일 다른 옷을 조합하여 입는것을 좋아한다.
예를 들어 코니가 가진 옷이 아래와 같고, 오늘 코니가 동그란 안경, 긴 코트, 파란색 티셔츠를 입었다면 다음날은 청바지를 추가로 입거나 동그란 안경 대신 검정 선글라스를 착용하거나 해야한다.                                                                          

 종류          이름
얼굴	동그란 안경, 검정 선글라스
상의	파란색 티셔츠
하의	청바지
겉옷	긴 코트

코니는 각 종류별로 최대 1가지 의상만 착용할 수 있다. 예를 들어 위 예시의 경우 동그란 안경과 검정 선글라스를 동시에 착용할 수는 없다.
착용한 의상의 일부가 겹치더라도, 다른 의상이 겹치지 않거나, 혹은 의상을 추가로 더 착용한 경우에는 서로 다른 방법으로 옷을 착용한 것으로 계산한다.
코니는 하루에 최소 한 개의 의상은 입는다.
코니가 가진 의상들이 담긴 2차원 배열 clothes가 주어질 때 서로 다른 옷의 조합의 수를 반환하도록 solution 함수를 작성해라.'''

#2) 제한 사항

'''clothes의 각 행은 [의상의 이름, 의상의 종류]로 이루어져 있다.
코니가 가진 의상의 수는 1개 이상 30개 이하이다.
같은 이름을 가진 의상은 존재하지 않는다.
clothes의 모든 원소는 문자열로 이루어져 있다.
모든 문자열의 길이는 1 이상 20 이하인 자연수이고 알파벳 소문자 또는 '_' 로만 이루어져 있다.'''

#3) 입출력 예시 

'''입출력 예 #1

colthes가 [["yellow_hat", "headgear"], ["blue_sunglasses", "eyewear"], ["green_turban", "headgear"]]이면, headgear에 해당하는 의상이 yellow_hat, green_turban이고 eyewear에 해당하는 의상이 blue_sunglasses이므로 아래와 같이 5개의 조합이 가능하다.

1. yellow_hat
2. blue_sunglasses
3. green_turban
4. yellow_hat + blue_sunglasses
5. green_turban + blue_sunglasses'''
    
#4) 코드 설명

'''solution 함수에서 colthes의 값을 입력 받는다.
경우의 수를 저장할 변수를 0으로 초기화한다.
의상을 종류별로 분류하기 위해 categorized_items라는 빈 딕셔너리를 생성한다.
반복문에서 item,과 category에 각각 colthes라는 리스트 안에 있는 리스트의 요소 값을 집어넣는다.
반복문 안에서는 만약 category가 categorized_item에 없으면 해당 카테고리가 없다는 뜻이므로 해당 카테고리를 추가해준다.
그렇지 않다면 categorize_item에 해당 카테고리에 value 값에 item을 추가하여 해당 카테고리에 의상을 추가한다.
반복문이 끝나면 a에 각 카테고리와 그에 속한 의상 리스트가 튜플 형태로 리스트를 저장한다.
answer은 a에 첫번째의 두번째 요소의 길이 + 1 한 값을 저장하여 맨 첫 카테고리의 의상 선택이 안된 경우의 수를 포함하여 모든 경우의 수로 지정하여 초기 값을 정한다.
반복문을 실행하여 i에 1부터 a의 길이빼기 1까지 i에 할당한다.
반복문 안에서는 answer의 값에 a에 i+1번째의 2번째 요소의 길이 +1한 값을 곱해서 answer에 다시 저장하여 각 카테고리의 의상 개수에 선택을 안한 경우를 포함한 현재까지의 모든 경우의 수를 구한다.
반복문이 끝나면 모든 카테고리에서 의상을 선택 안한 경우의 수 한 가지를 제외한 값을 반환한다.'''

#문제 출처 : 프로그래머스