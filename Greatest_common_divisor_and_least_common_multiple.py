def solution(n, m):
    answer = []
    for i in range(1,n+1):
        if n % i == 0 and m % i == 0:
            max1 = i
    answer.append(max1)
    answer.append(max1 * (n // max1) * (m // max1))
    return answer



#1) 문제 설명

'두 수를 입력받아 두 수의 최대공약수와 최소공배수를 반환하는 함수, solution을 완성해라. 배열의 맨 앞에 최대공약수, 그다음 최소공배수를 넣어 반환하면 된다. 예를 들어 두 수 3, 12의 최대공약수는 3, 최소공배수는 12이므로 solution(3, 12)는 [3, 12]를 반환해야 한다.'

#2) 제한 사항

'두 수는 1이상 1000000이하의 자연수입니다.'

#3) 입출력 예시

'''입출력 예 #1

n이 3이고 m이 12면 [3,12]를 반환한다.
입출력 예 #2

자연수 2와 5의 최대공약수는 1, 최소공배수는 10이므로 [1, 10]을 리턴해야 한다.'''

#4) 코드 설명

'''solution 함수에서 n과 m의 값을 입력 받는다.
최대 공약수와 최소 공배수를 저장할 빈 리스트를 만든다.
반복문을 통해 1부터 n까지의 값을 하나씩 i에 대입한다.
반복문을 하나씩 실행할 때, 만약 n의 값을 i로 나누었을 때의 나머지와 m의 값을 i로 나누었을 때의 나머지 모두 0이라면 최대 공약수는 그 때의 i가 된다.
최대 공약수를 구했으면 그 값을 이용해 n과 m을 각각 나누고 그 때의 몫들과 최대 공약수를 곱하게 되면 최소 공배수가 된다.
이 값들을 모두 answer 리스트에 집어 넣어 반환한다.'''

#문제 출처 : 프로그래머스