def solution(brown, yellow):
    total = brown + yellow
    for height in range(1, total + 1):
        if total % height == 0:
            width = total // height
            if width >= height and (width - 2) * (height - 2) == yellow:
                return [width, height]


# 1) 문제 설명

"""Leo는 카펫을 사러 갔다가 아래 그림과 같이 중앙에는 노란색으로 칠해져 있고 테두리 1줄은 갈색으로 칠해져 있는 격자 모양 카펫을 봤다.

Leo는 집으로 돌아와서 아까 본 카펫의 노란색과 갈색으로 색칠된 격자의 개수는 기억했지만, 전체 카펫의 크기는 기억하지 못했다.
Leo가 본 카펫에서 갈색 격자의 수 brown, 노란색 격자의 수 yellow가 매개변수로 주어질 때 카펫의 가로, 세로 크기를 순서대로 배열에 담아 반환하도록 solution 함수를 작성해라."""

# 2) 제한 사항

"""갈색 격자의 수 brown은 8 이상 5,000 이하인 자연수이다.
노란색 격자의 수 yellow는 1 이상 2,000,000 이하인 자연수이다.
카펫의 가로 길이는 세로 길이와 같거나, 세로 길이보다 길다."""


# 3) 입출력 예시

"""입출력 예 #1

brown이 10, yellow가 2이면, [4,3]을 반환한다.



입출력 예 #2

brown이 8, yellow가 1이면, [3,3]을 반환한다.



입출력 예 #3

brown이 24, yellow가 24이면, [8,6]을 반환한다."""

# 4) 코드 설명

"""solution 함수에서 brown, yellow의 값을 입력 받는다.
총 크기를 구하기 위해 total에 brown과 yellow을 더한 값을 저장한다.
반복문을 통해 height에 1부터 total 값까지 1씩 증가시키면 집어넣는다.
반복문 안에서는 만약 total 값 나누기 height한 값이 0이라면 width에 total값을 height로 나눈 몫을 저장한다.
그 후 만약 width가 height보다 크거나 같고 (width-2) * (height - 2)한 값이 yellow 값과 같다면 , 즉 가로 길이가 세로보다 크거나 같고 yellow값과 일치한다면 width와 height값을 리스트 형태로 반환한다."""

# 문제 출처 : 프로그래머스
