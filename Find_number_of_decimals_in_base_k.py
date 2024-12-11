# 소수 판별 함수
def is_prime(num):
    # 1 이하의 숫자는 소수가 아님
    if num <= 1:
        return False
    # 2부터 num의 제곱근까지 나누어 보며, 나누어 떨어지면 소수가 아님
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    # 나누어 떨어지지 않으면 소수
    return True


def solution(n, k):
    # n을 k진수로 변환할 결과를 저장할 빈 리스트
    k_base = []

    # n이 k보다 클 때까지 반복하여 k진수로 변환
    while n >= k:
        # n을 k로 나눈 나머지를 리스트에 추가
        k_base.append(str(n % k))
        # n을 k로 나눈 몫으로 n을 갱신
        n = n // k

    # 마지막 n 값을 리스트에 추가
    k_base.append(str(n))
    # 리스트를 뒤집어서 k진수 문자열을 얻음
    k_base.reverse()
    # 리스트를 하나의 문자열로 합침
    k_base = "".join(k_base)

    # k진수 문자열에서 '0'을 기준으로 자른 부분들을 potential_primes에 저장
    potential_primes = k_base.split("0")

    # 소수를 셀 변수 초기화
    count = 0
    # potential_primes에서 각 부분을 소수 판별
    for potential_prime in potential_primes:
        # 빈 문자열이 아니고, 소수라면 count 증가
        if potential_prime != "" and is_prime(int(potential_prime)):
            count += 1

    # 소수의 개수를 반환
    return count


# 1) 문제 설명

"""양의 정수 n이 주어집니다. 이 숫자를 k진수로 바꿨을 때, 변환된 수 안에 아래 조건에 맞는 소수(Prime number)가 몇 개인지 알아보려 한다.
0P0처럼 소수 양쪽에 0이 있는 경우
P0처럼 소수 오른쪽에만 0이 있고 왼쪽에는 아무것도 없는 경우
0P처럼 소수 왼쪽에만 0이 있고 오른쪽에는 아무것도 없는 경우
P처럼 소수 양쪽에 아무것도 없는 경우단, P는 각 자릿수에 0을 포함하지 않는 소수이다.
예를 들어, 101은 P가 될 수 없다.
예를 들어, 437674을 3진수로 바꾸면 211020101011이다. 여기서 찾을 수 있는 조건에 맞는 소수는 왼쪽부터 순서대로 211, 2, 11이 있으며, 총 3개이다. (211, 2, 11을 k진법으로 보았을 때가 아닌, 10진법으로 보았을 때 소수여야 한다는 점에 주의한다.) 211은 P0 형태에서 찾을 수 있으며, 2는 0P0에서, 11은 0P에서 찾을 수 있다.
정수 n과 k가 매개변수로 주어진다. n을 k진수로 바꿨을 때, 변환된 수 안에서 찾을 수 있는 위 조건에 맞는 소수의 개수를 반환하도록 solution 함수를 완성해라."""


# 2) 제한 사항

"""1 ≤ n ≤ 1,000,000
3 ≤ k ≤ 10"""


# 3) 입출력 예시


# 입출력 예 #1


"""n이 110011이고 k가 10이면, 110011을 10진수로 바꾸면 110011이다. 여기서 찾을 수 있는 조건에 맞는 소수는 11, 11 2개이다. 이와 같이, 중복되는 소수를 발견하더라도 모두 따로 세어야 한다."""

# 4) 코드 설명

"""소수를 판별하기 위해 is_prime이라는 함수를 만든다.
그 함수 안에서는 만약 입력받은 num의 값이 1보다 작거나 같다면, 1이하는 소수가 아니므로 False를 반환한다.
그렇지 않다면, for문을 사용하여 2부터 num의 제곱근까지 값을 하나씩 i에 집어넣는다.
반복문 안에서는 num을 i로 나누었을 때 나머지가 0이라면 1과 자기 자신 이외에도 약수를 가진다는 뜻이므로 소수가 아닌게 되어 False를 반환한다.
함수 안에 모든 동작을 실행하여도 아무 문제가 없었다면 그 숫자는 소수이므로 True를 반환한다.
solution 함수에 n, k의 값을 입력 받는다.
n을 k 진수로 변환할 결과를 저장하기 위해 k_base라는 빈 리스트를 만든다.
while문을 통해 n의 값이 k보다 클때까지 반복한다.
while문 안에서는 k_base에 n을 k로 나눈 나머지를 추가한다.
그 후 n의 값을 k로 나눈 몫으로 n을 갱신한다.
while문이 끝나면 마지막 n의 값을 리스트에 추가하고 뒤집은 후 join 함수를 이용해 하나의 문자열로 합쳐 k진수로 변환을 완료한다.
그 후 split 함수를 이용해 0을 기준으로 자르고 potential_primes에 저장하여 소수가 될 수 있는 숫자를 분류한다.
다음으로 count의 값을 0으로 초기화하여 소수를 셀 준비를 한다.
for문을 통해 pornetial_priems에 들어 있는 소수가 될 가능성이 있는 숫자들을 하나씩 potential_prime에 집어넣는다.
반복문 안에서는 만약 potential_prime의 값이 비어있지 않고 위에서 만든 소수인지 확인하는 함수를 통해 potential_prime의 값이 소수라면 count의 값을 1 증가시킨다.
반복문이 끝나면 그 때의 count 값을 반환한다."""

# 문제 출처 : 프로그래머스
