def solution(t, p):
    answer = 0
    for i in range(len(t) - len(p) + 1):
        if int(t[i : i + len(p)]) <= int(p):
            answer += 1
    return answer


# 1) 문제 설명

"""숫자로 이루어진 문자열 t와 p가 주어질 때, t에서 p와 길이가 같은 부분문자열 중에서, 이 부분문자열이 나타내는 수가 p가 나타내는 수보다 작거나 같은 것이 나오는 횟수를 return하는 함수 solution을 완성하라.
예를 들어, t="3141592"이고 p="271" 인 경우, t의 길이가 3인 부분 문자열은 314, 141, 415, 159, 592이다. 이 문자열이 나타내는 수 중 271보다 작거나 같은 수는 141, 159 2개이다."""

# 2) 제한 사항

"1 ≤ p의 길이 ≤ 18"
"p의 길이 ≤ t의 길이 ≤ 10,000"
"t와 p는 숫자로만 이루어진 문자열이며, 0으로 시작하지 않는다."

# 3) 입출력 예시

"입출력 예 #1"

"""p의 길이가 1이므로 t의 부분문자열은 "5", "0", 0", "2", "2", "0", "8", "3", "9", "8", "7", "8"이며 이중 7보다 작거나 같은 숫자는 "5", "0", "0", "2", "2", "0", "3", "7" 이렇게 8개가 있다."""

"입출력 예 #2"

"""p의 길이가 2이므로 t의 부분문자열은 "10", "02", "20", "03"이며, 이중 15보다 작거나 같은 숫자는 "10", "02", "03" 이렇게 3개이다. "02"와 "03"은 각각 2, 3에 해당한다는 점에 주의하라."""

# 4). 코드 설명

"""solution 함수에서 t와 p의 값을 입력 받는다.
p보다 작거나 같은 숫자를 세기 위해 answer은 초기 값을 0으로 준다.
반복문을 통해 앞에서부터 p의 길이 만큼 나누며 비교하기 위해 t의 길이에서 p의 길이를 빼주고 0부터 시작하기 때문에 그 값에 1을 더한만큼 반복한다.
만약 문자열 t의 i번쨰부터 p의 길이만큼 나눈 값을 정수로 바꿨을 때, 문자열 p를 정수로 바꾼 값보다 적거나 같다면 answer에 값을 1 추가한다.
반복문이 끝나면 p보다 작거나 같은 값이 모두 구해진 것이므로 answer을 반환한다.
문제 출처 : 프로그래머스"""
