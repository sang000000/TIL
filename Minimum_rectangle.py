def solution(sizes):
    for i in range(len(sizes)):
        sizes[i].sort(reverse=True)
    weight = max(sizes)[0]
    for i in range(len(sizes)):
        sizes[i].sort()
    high = max(sizes)[0]
    answer = weight * high
    return answer


# 1) 문제 설명

"""명함 지갑을 만드는 회사에서 지갑의 크기를 정하려고 한다. 다양한 모양과 크기의 명함들을 모두 수납할 수 있으면서, 작아서 들고 다니기 편한 지갑을 만들어야 한다. 이러한 요건을 만족하는 지갑을 만들기 위해 디자인팀은 모든 명함의 가로 길이와 세로 길이를 조사했다.
아래 표는 4가지 명함의 가로 길이와 세로 길이를 나타낸다."""


"""명함 번호 가로 길이 세로 길이
    1	      60	  50
    2	      30	  70
    3	      60	  30
    4	      80	  40       """

"""가장 긴 가로 길이와 세로 길이가 각각 80, 70이기 때문에 80(가로) x 70(세로) 크기의 지갑을 만들면 모든 명함들을 수납할 수 있다. 하지만 2번 명함을 가로로 눕혀 수납한다면 80(가로) x 50(세로) 크기의 지갑으로 모든 명함들을 수납할 수 있다. 이때의 지갑 크기는 4000(=80 x 50)이다.
모든 명함의 가로 길이와 세로 길이를 나타내는 2차원 배열 sizes가 매개변수로 주어진다. 모든 명함을 수납할 수 있는 가장 작은 지갑을 만들 때, 지갑의 크기를 return 하도록 solution 함수를 완성해라."""

# 2) 제한 사항

"""sizes의 길이는 1 이상 10,000 이하이다.
sizes의 원소는 [w, h] 형식이다.
w는 명함의 가로 길이를 나타낸다.
h는 명함의 세로 길이를 나타낸다.
w와 h는 1 이상 1,000 이하인 자연수이다."""

# 3) 입출력 예시

"""입출력 예 #1

sizes가 [[10, 7], [12, 3], [8, 15], [14, 7], [5, 15]]이면, 명함들을 적절히 회전시켜 겹쳤을 때, 3번째 명함(가로: 8, 세로: 15)이 다른 모든 명함보다 크기가 크다. 따라서 지갑의 크기는 3번째 명함의 크기와 같으며, 120(=8 x 15)을 return 한다.
입출력 예 #2

sizes가 [[14, 4], [19, 6], [6, 16], [18, 7], [7, 11]]이면, 명함들을 적절히 회전시켜 겹쳤을 때, 모든 명함을 포함하는 가장 작은 지갑의 크기는 133(=19 x 7)이다."""

# 4). 코드 설명

"""solution 함수에서 sizes의 값을 입력 받는다.
반복문을 통해 리스트 sizes의 길이만큼 리스트 sizes 안에 있는 리스트의 요소들을 각각 오름차순으로 바꿔준다.
리스트 sizes 안에 있는 리스트의 요소들 중 최고 값을 찾아 가로 길이로 정한다.
다시 한번 반복문을 통해 리스트 sizes의 길이만큼 리스트 sizes 안에 있는 리스트의 요소들을 각각 내림차순으로 바꿔준다.
그 후 리스트 sizes 안에 있는 리스트의 요소들 중 최고 값을 찾아 세로 길이로 정한다.
마지막으로 설정한 가로 길이와 세로 길이를 곱해서 크기를 구하여 반환한다.

출처: 프로그래머스"""
