def solution(n):
    reversed_n = ''
    answer = 0
    while n >= 3 :
        reversed_n += str(n % 3)
        n = n //3

    reversed_n += str(n)
    for i in range(len(reversed_n)):
        answer += 3**i * int(reversed_n[-(i+1)])
    return answer




# 1) 문제 설명

'자연수 n이 매개변수로 주어집니다. n을 3진법 상에서 앞뒤로 뒤집은 후, 이를 다시 10진법으로 표현한 수를 return 하도록 solution 함수를 완성해라.'

#2) 제한 사항

'n은 1 이상 100,000,000 이하인 자연수이다.'

#3) 입출력 예시

'''입출력 예 #1

답을 도출하는 과정은 다음과 같다.
n (10진법)                                       n (3진법)                                       앞뒤 반전(3진법)                10진법으로 표현
45	1200	0021	7
따라서 7을 return 해야 한다.
입출력 예 #2

답을 도출하는 과정은 다음과 같다.
n (10진법)                                      n (3진법)                                앞뒤 반전(3진법)               10진법으로 표현
125	11122	22111	229
따라서 229를 return 해야 한다.'''


#4) 코드 설명

'''solution 함수에서 n의 값을 입력 받는다.
앞뒤가 반전 된 n(3진법)을 저장하기 위해 빈 문자열을 생성한다.
앞뒤가 반전 된 n(3진법)을 10진수로 변한 값을 계산하기 위해 answer을 초기 값을 0으로 한다.
n이 3보다 작을 때까지 반복문을 통해 reversed_n에 n 나누기 3을 한 나머지 값을 문자열로 바꿔서 집어 넣는다.
그 후 n의 값을 n 나누기 3한 값의 몫으로 바꾼다.
n의 값이 최종적으로 3보다 작아지면 반복문은 종료된다.
그 후 reversed_n에 n의 값을 넣어 앞뒤가 반전 된 n(3진법)을 완성한다.
그 다음에 반복문을 통해 reversed_n의 길이만큼 0부터 길이 빼기 1까지 반복한다.
반복문 안에서는 answer에 3의 i 제곱한 값에 reversed_n의 뒤에서 부터 문자열로 바꾼 상태로 가져와 곱해서 더한 후 그 값을 반환한다.'''
#문제 출처 : 프로그래머스
