def solution(babbling):
    answer = 0
    a = ["aya", "ye", "woo", "ma"]

    for i in babbling:
        for j in a:
            if j * 2 not in i:
                i = i.replace(j, " ")

        if i.isspace():
            answer += 1

    return answer


# 1) 문제 설명

"""머쓱이는 태어난 지 11개월 된 조카를 돌보고 있다. 조카는 아직 "aya", "ye", "woo", "ma" 네 가지 발음과 네 가지 발음을 조합해서 만들 수 있는 발음밖에 하지 못하고 연속해서 같은 발음을 하는 것을 어려워한다. 문자열 배열 babbling이 매개변수로 주어질 때, 머쓱이의 조카가 발음할 수 있는 단어의 개수를 return하도록 solution 함수를 완성해라."""

# 2) 제한 사항

"""1 ≤ babbling의 길이 ≤ 100
1 ≤ babbling[i]의 길이 ≤ 30
문자열은 알파벳 소문자로만 이루어져 있다."""

# 3) 입출력 예시


"""입출력 예 #1

babbling이 ["aya", "yee", "u", "maa"]이면 발음할 수 있는 것은 "aya"뿐이다. 따라서 1을 반환한다.


입출력 예 #2

babbling이 ["ayaye", "uuu", "yeye", "yemawoo", "ayaayaa"]이면 발음할 수 있는 것은 "aya" + "ye" = "ayaye", "ye" + "ma" + "woo" = "yemawoo"로 2개이다. "yeye"는 같은 발음이 연속되므로 발음할 수 없다. 따라서 2를 반환한다."""


# 4) 코드 설명

"""solution 함수에서 babbling의 값을 입력 받는다.
단어의 개수를 세기 위해 answer을 초기 값 0으로 준다.
발음 할 수 있는 단어를 비교하기 위해 [ "aya", "ye", "woo", "ma"] 리스트를 만든다.
반복문을 통해 i에 입력 받은 babbling 리스트의 요소 값을 하나씩 넣는다.
반복문 안에서 리스트 a의 요소 값을 j에 하나씩 집어넣는다.
문제에서 연속으로 발음할 수 없다고 했으므로 만약 j 값 곱하기 2한 값이 i에 없다면, 즉 똑같은 단어가 연속으로 구성된 값이 i에 없다면 해당 j값을 공백으로 대체한다.
만약 i가 공백을 가지고 있다면 모두 공백이라면, 발음이 가능한 것이므로 answer의 값을 1 증가시킨다.
반복문이 끝나면 answer을 반환한다."""

# 문제 출처 : 프로그래머스
