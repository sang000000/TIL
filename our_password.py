def solution(s, skip, index):
    answer = ""
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    for char in skip:
        alphabet = alphabet.replace(char, "")

    for j in s:
        current_index = alphabet.index(j)
        new_index = (current_index + index) % len(alphabet)
        answer += alphabet[new_index]

    return answer


# 1) 문제 설명

"""두 문자열 s와 skip, 그리고 자연수 index가 주어질 때, 다음 규칙에 따라 문자열을 만들려 한다. 암호의 규칙은 다음과 같다.
문자열 s의 각 알파벳을 index만큼 뒤의 알파벳으로 바꿔준다.
index만큼의 뒤의 알파벳이 z를 넘어갈 경우 다시 a로 돌아간다.
skip에 있는 알파벳은 제외하고 건너뛴다.
예를 들어 s = "aukks", skip = "wbqd", index = 5일 때, a에서 5만큼 뒤에 있는 알파벳은 f지만 [b, c, d, e, f]에서 'b'와 'd'는 skip에 포함되므로 세지 않는다. 따라서 'b', 'd'를 제외하고 'a'에서 5만큼 뒤에 있는 알파벳은 [c, e, f, g, h] 순서에 의해 'h'가 된다. 나머지 "ukks" 또한 위 규칙대로 바꾸면 "appy"가 되며 결과는 "happy"가 된다.
두 문자열 s와 skip, 그리고 자연수 index가 매개변수로 주어질 때 위 규칙대로 s를 변환한 결과를 return하도록 solution 함수를 완성해라.
"""
# 2) 제한 사항

"""5 ≤ s의 길이 ≤ 50
1 ≤ skip의 길이 ≤ 10
s와 skip은 알파벳 소문자로만 이루어져 있다.
skip에 포함되는 알파벳은 s에 포함되지 않는다.
1 ≤ index ≤ 20"""

# 3) 입출력 예시

"""입출력 예 #1

s가 "aukks"이고 skip가 "wbqd"이고 index가 5이면, "happy"가 반환된다."""

# 4) 코드 설명

"""solution 함수에서 s, skip, index의 값을 입력 받는다.
정답을 저장하기 위해 answer을 빈 리스트로 초기값을 설정한다.
알파벳 건너 뛰기 위해 aphabet에 a부터z까지 입력하여 저장한다.﻿
반복문을 통해 skip에 값을 char에 집어넣어 skip에 있는 요소 값들을 alphabet에서 제거하여 alphabet에 저장한다.
위에 반복문이 끝나고 나서 다른 반복문을 통해 s의 값을 j에 하나씩 집어넣는다.
반복문 안에서는 index 함수를 이용해서 해당 j의 값의 위치의 인덱스 값을 current_index에 저장한다.
그 후 current_index 더하기 index를 한 값을 alphabet의 길이로 나눈 나머지 값을 new_index에 저장한다.
그 후 answer에 alphabet에서 new_indext 값을 집어넣어 규칙에 따라 바뀐 알파벳을 저장한다.
반복문이 끝나면 answer을 반환한다."""

# 문제 출처 : 프로그래머스
