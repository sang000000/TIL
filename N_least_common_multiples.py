import math
from functools import reduce

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

def solution(arr):
    return reduce(lcm, arr)



#1) 문제 설명

'''두 수의 최소공배수(Least Common Multiple)란 입력된 두 수의 배수 중 공통이 되는 가장 작은 숫자를 의미한다. 예를 들어 2와 7의 최소공배수는 14가 된다. 정의를 확장해서, n개의 수의 최소공배수는 n 개의 수들의 배수 중 공통이 되는 가장 작은 숫자가 된다. n개의 숫자를 담은 배열 arr이 입력되었을 때 이 수들의 최소공배수를 반환하는 함수, solution을 완성해라.
'''
#2) 제한 사항

'''arr은 길이 1이상, 15이하인 배열이다.
arr의 원소는 100 이하인 자연수이다.'''


#3) 입출력 예시 

'''입출력 예 #1

arr이 [2,6,8,14]이면 168이 반환된다.'''

#4) 코드 설명

'''최대 공약수를 구하기 위해 수학 관련 함수를 제공하는 math 라이브러리를 불러온다.
최대 공배수를 구하기 위해 배열의 요소를 누적하여 하나의 결과로 만드는 reduce함수를 가져온다.
최대 공약수 함수인 lcm을 만들어 a,b의 값을 가져온다.
함수에 안에서는 a와 b의 값을 가져와 abs 함수로 절대값으로 취하고 이 값을 gcd 함수를 이용해 최대 공약수로 나눈  몫을 반환한다.
solution 함수에서 arr의 값을 가져온다.
reduce 함수를 사용하여 배열의 첫 번째 요소와 두 번째 요소의 최대 공배수를 계산한 후, 그 결과를 세 번째 요소와 L최대공약수를 계산하는 방식으로 배열의 모든 요소에 대해 LCM 함수를 반복적으로 적용하고 그 값을 반환한다.
'''
#문제 출처 : 프로그래머스