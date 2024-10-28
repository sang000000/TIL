def solution(number, limit, power):
    m = []
    for i in range(1,number+1):
        count = 0
        for k in range(1,int(i ** (1/2))+1):
            if i % k == 0:
                count += 1
                if k**2 != i:                 
                    count += 1
            if count > limit:               
                count = power
                break
        m.append(count)
    return sum(m)


#1) 문제 설명

'''숫자나라 기사단의 각 기사에게는 1번부터 number까지 번호가 지정되어 있다. 기사들은 무기점에서 무기를 구매하려고 한다.
각 기사는 자신의 기사 번호의 약수 개수에 해당하는 공격력을 가진 무기를 구매하려 한다. 단, 이웃나라와의 협약에 의해 공격력의 제한수치를 정하고, 제한수치보다 큰 공격력을 가진 무기를 구매해야 하는 기사는 협약기관에서 정한 공격력을 가지는 무기를 구매해야 한다.
예를 들어, 15번으로 지정된 기사단원은 15의 약수가 1, 3, 5, 15로 4개 이므로, 공격력이 4인 무기를 구매한다. 만약, 이웃나라와의 협약으로 정해진 공격력의 제한수치가 3이고 제한수치를 초과한 기사가 사용할 무기의 공격력이 2라면, 15번으로 지정된 기사단원은 무기점에서 공격력이 2인 무기를 구매한다. 무기를 만들 때, 무기의 공격력 1당 1kg의 철이 필요하다. 그래서 무기점에서 무기를 모두 만들기 위해 필요한 철의 무게를 미리 계산하려 한다.각 기사는 자신의 기사 번호의 약수 개수에 해당하는 공격력을 가진 무기를 구매하려 한다. 단, 이웃나라와의 협약에 의해 공격력의 제한수치를 정하고, 제한수치보다 큰 공격력을 가진 무기를 구매해야 하는 기사는 협약기관에서 정한 공격력을 가지는 무기를 구매해야 한다.
기사단원의 수를 나타내는 정수 number와 이웃나라와 협약으로 정해진 공격력의 제한수치를 나타내는 정수 limit와 제한수치를 초과한 기사가 사용할 무기의 공격력을 나타내는 정수 power가 주어졌을 때, 무기점의 주인이 무기를 모두 만들기 위해 필요한 철의 무게를 return 하는 solution 함수를 완성하라.
'''
#2) 제한 사항

'''1 ≤ number ≤ 100,000
2 ≤ limit ≤ 100
1 ≤ power ≤ limit'''

#3) 입출력 예시



'''입출력 예 #1

number가 5이고 limit가 3이고 power가 2이면, 1부터 5까지의 약수의 개수는 순서대로 [1, 2, 2, 3, 2]개이다. 모두 공격력 제한 수치인 3을 넘지 않기 때문에 필요한 철의 무게는 해당 수들의 합인 10이 된다. 따라서 10을 return 한다.


입출력 예 #2

number가 10이고 limit가 3이고 power가 2이면, 1부터 10까지의 약수의 개수는 순서대로 [1, 2, 2, 3, 2, 4, 2, 4, 3, 4]개입니다. 공격력의 제한수치가 3이기 때문에, 6, 8, 10번 기사는 공격력이 2인 무기를 구매한다. 따라서 해당 수들의 합인 21을 return 한다.'''

#4) 코드 설명

'''solution 함수에서 number과 limit와 power의 값을 입력 받는다.
각 약수의 개수들의 합을 구하기 위해 m이라는 빈 리스트를 만든다.﻿
반복문을 이용하여 1부터 number까지 1씩 증가시키며 반복한다.
반복문 안에서는 각 i 값들의 약수의 개수를 세기 위해 매번 count를 0으로 초기화 한다.
반복문 안에 반복문에서는 1부터 i의 제곱근가지 1씩 증가시키면서 만약 i값을 k로 나눴을 떄 값이 0이라면, 약수이므로﻿ count의 값을 1 증가시킨다.
그 다음에 만약 k를 제곱한 값이 i와 다르다면 count 값을 1 증가시켜 제곱이 되는 수를 중복 방지해준다/
반복문 안에 반복문이 끝나면 coutn의 값을 리스트 m에 추가한다.
모든 반복문이 끝나면 리스트 m의 합계를 반환한다.'''

#문제 출처 : 프로그래머스