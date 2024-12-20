def solution(n):
    MOD = 1234567
    dp = [0] * (n + 1)

    dp[1] = 1
    if n >= 2:
        dp[2] = 2

    for i in range(3, n + 1):
        dp[i] = (dp[i - 1] + dp[i - 2]) % MOD

    return dp[n]


# 1) 문제 설명

"""효진이는 멀리 뛰기를 연습하고 있다. 효진이는 한번에 1칸, 또는 2칸을 뛸 수 있다. 칸이 총 4개 있을 때, 효진이는
(1칸, 1칸, 1칸, 1칸)
(1칸, 2칸, 1칸)
(1칸, 1칸, 2칸)
(2칸, 1칸, 1칸)
(2칸, 2칸)
의 5가지 방법으로 맨 끝 칸에 도달할 수 있다. 멀리뛰기에 사용될 칸의 수 n이 주어질 때, 효진이가 끝에 도달하는 방법이 몇 가지인지 알아내, 여기에 1234567를 나눈 나머지를 리턴하는 함수, solution을 완성하라. 예를 들어 4가 입력된다면, 5를 return하면 된다."""
# 2) 제한 사항

"""n은 1 이상, 2000 이하인 정수이다."""


# 3) 입출력 예시


"""입출력 예 #1

n이 3이면

(2칸, 1칸)

(1칸, 2칸)
(1칸, 1칸, 1칸)
총 3가지 방법으로 멀리 뛸 수 있다."""


# 4) 코드 설명

"""solution 함수에서 n의 값을 입력 받는다.
1234567로 나눈 나머지를 구하기 위해 MOD에 저장한다.
길이 ( n + 1 )인 리스트 dp를 생성한다.
1칸에 도달하는 방법은 1가지이므로 dp[1]은 1로 설정한다.
만약 n이 2이상이라면, 2칸에 도달하는 방법은 2가지이므로, dp[2]는 2로 설정한다.
반복문을 사용하여 i에 3부터 n까지 값을 하나씩 집어넣는다.
반복문 안에서는 n칸까지 가는 경우의 수는 n-1칸의 경우의 수 더하기 n-2칸의 경우의 수 값과 같으므로 dp[i-1] + df[i-2]를 더한 값에 MOD를 나누고 그 값을 dp[i]에 저장한다.
반복문이 끝나면 n칸의 경우의 수를 출력한다."""

# 문제 출처 : 프로그래머스
