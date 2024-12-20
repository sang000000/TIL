def solution(n):
    a = [0, 1]
    for i in range(2, n + 1):
        b = a[0] + a[1]
        a[0], a[1] = a[1], b
    answer = b % 1234567

    return answer


# 1) 문제 설명

"""피보나치 수는 F(0) = 0, F(1) = 1일 때, 1 이상의 n에 대하여 F(n) = F(n-1) + F(n-2) 가 적용되는 수이다.
예를들어
F(2) = F(0) + F(1) = 0 + 1 = 1
F(3) = F(1) + F(2) = 1 + 1 = 2
F(4) = F(2) + F(3) = 1 + 2 = 3
F(5) = F(3) + F(4) = 2 + 3 = 5
와 같이 이어진다.
2 이상의 n이 입력되었을 때, n번째 피보나치 수를 1234567으로 나눈 나머지를 반환하는 함수, solution을 완성해라."""


# 2) 제한 사항

"""n은 2 이상 100,000 이하인 자연수이다."""


# 3) 입출력 예시


"""입출력 예 #1



n이 3이면, F(3)은 2이므로 1234567으로 나눴을 때 나머지가 2가 나온다. 따라서 2를 반환한다."""


# 4) 코드 설명

"""solution 함수에서 n의 값을 입력 받는다.
피보나치 수를 구하기 위해 a에 초기 값 [0,1]로 두어 n의 두 번째 전 값과 하나 전 값을 저장하는 용도로 사용한다.
n의 피보나치 값을 구하기 위해 반복문을 돌려 2부터 n까지 반복한다.
반복문 안에서는 b에 a[0]+ a[1]의 값을 해서 i 번째에 해당하는 피보나치 수를 구하여 저장한다.
그후 a[0]는 a[1]의 값으로 바꾸고 a[1]은 b의 값으로 바꿔서 다음에 해당하는 피보나치 수를 구하기 위한 준비를 한다.
모든 반복문이 끝나면 b에는 n번째의 피보나치 수가 구해지므로 이 값에 1234567을 나누고 나머지 값을 answer에 저장한다.
그 후 answer 값을 반환한다."""

# 문제 출처 : 프로그래머스
