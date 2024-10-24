def solution(k, m, score):
    score.sort(reverse = True)
    answer = 0
    for i in range(0,len(score)-m,m):
        answer += min(score[i:i+m]) * m
    if len(score) % m == 0:
        answer += min(score[-m:]) * m       
    return answer




#1) 문제 설명

'''과일 장수가 사과 상자를 포장하고 있다. 사과는 상태에 따라 1점부터 k점까지의 점수로 분류하며, k점이 최상품의 사과이고 1점이 최하품의 사과이다. 사과 한 상자의 가격은 다음과 같이 결정된다.
한 상자에 사과를 m개씩 담아 포장한다.
상자에 담긴 사과 중 가장 낮은 점수가 p (1 ≤ p ≤ k)점인 경우, 사과 한 상자의 가격은 p * m 이다. 
과일 장수가 가능한 많은 사과를 팔았을 때, 얻을 수 있는 최대 이익을 계산하고자 합니다.(사과는 상자 단위로만 판매하며, 남는 사과는 버린다
예를 들어, k = 3, m = 4, 사과 7개의 점수가 [1, 2, 3, 1, 2, 3, 1]이라면, 다음과 같이 [2, 3, 2, 3]으로 구성된 사과 상자 1개를 만들어 판매하여 최대 이익을 얻을 수 있다.
(최저 사과 점수) x (한 상자에 담긴 사과 개수) x (상자의 개수) = 2 x 4 x 1 = 8
사과의 최대 점수 k, 한 상자에 들어가는 사과의 수 m, 사과들의 점수 score가 주어졌을 때, 과일 장수가 얻을 수 있는 최대 이익을 return하는 solution 함수를 완성해라.'''

#2) 제한 사항

'''3 ≤ k ≤ 9
3 ≤ m ≤ 10
7 ≤ score의 길이 ≤ 1,000,000
1 ≤ score[i] ≤ k
이익이 발생하지 않는 경우에는 0을 return 해라.'''

#3) 입출력 예시

'''입출력 예 #1

k가 4이고 m이 3이고 score가 [4, 1, 2, 2, 4, 4, 4, 4, 1, 2, 4, 2]이면
다음과 같이 사과 상자를 포장하여 모두 팔면 최대 이익을 낼 수 있다.
사과 상자가격
[1, 1, 2]	1 x 3 = 3
[2, 2, 2]	2 x 3 = 6
[4, 4, 4]	4 x 3 = 12
[4, 4, 4]	4 x 3 = 12
따라서 (1 x 3 x 1) + (2 x 3 x 1) + (4 x 3 x 2) = 33을 return한다.'''

#4) 코드 설명

'''solution 함수에서 k와 m과 score의 값을 입력 받는다.
최대 이익을 구하기 위해 score를 sort 함수를 이용해 내림차순으로 정리한다.
answer에 최대 이익 값을 구하기 위해 초기 값을 0으로 준다.
반복문을 이용하여 m개씩 나눌 것이므로 0부터 score의 길이 빼기 m개씩 증가하여 i에 집어넣는다.
answer은 score 리스트를 i부터 i+m 전까지 나눈 값 중에 최소를 찾아 m이랑 곱한 값을 더한다. 즉 인덱스를 이용하여 m의 개수만큼 score 리스트를 나누고 그 중 제일 작은 값을 개수랑 곱하여 한 박스에 이익을 구한다.
반복문이 끝나면 만약 score의 길이가 m으로 나눌떄 나머지가 0이라면, 즉 남는거 없이 딱 떨어지게 나눌 수 있다면, 반복문이 끝나고 남아 있는 것들도 위와 똑같이 이익을 구하여 더해준다.
그 후 값을 반환한다.'''

#문제 출처 : 프로그래머스

