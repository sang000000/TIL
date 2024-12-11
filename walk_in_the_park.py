def solution(park, routes):
    answer = []
    for i in range(len(park)):
        if "S" in park[i]:
            answer = [i, park[i].index("S")]
            break
    for j in routes:
        direction = j[0]
        distance = int(j[2])

        if direction == "E" and (answer[1] + distance) < len(park[0]):
            if all(
                park[answer[0]][answer[1] + k] != "X" for k in range(1, distance + 1)
            ):
                answer[1] += distance
        elif direction == "W" and (answer[1] - distance) >= 0:
            if all(
                park[answer[0]][answer[1] - k] != "X" for k in range(1, distance + 1)
            ):
                answer[1] -= distance
        elif direction == "S" and (answer[0] + distance) < len(park):
            if all(
                park[answer[0] + k][answer[1]] != "X" for k in range(1, distance + 1)
            ):
                answer[0] += distance
        elif direction == "N" and (answer[0] - distance) >= 0:
            if all(
                park[answer[0] - k][answer[1]] != "X" for k in range(1, distance + 1)
            ):
                answer[0] -= distance

    return answer


# 1) 문제 설명

"""지나다니는 길을 'O', 장애물을 'X'로 나타낸 직사각형 격자 모양의 공원에서 로봇 강아지가 산책을 하려한다. 산책은 로봇 강아지에 미리 입력된 명령에 따라 진행하며, 명령은 다음과 같은 형식으로 주어진다.
["방향 거리", "방향 거리" … ]
예를 들어 "E 5"는 로봇 강아지가 현재 위치에서 동쪽으로 5칸 이동했다는 의미이다. 로봇 강아지는 명령을 수행하기 전에 다음 두 가지를 먼저 확인한다.
주어진 방향으로 이동할 때 공원을 벗어나는지 확인한다.
주어진 방향으로 이동 중 장애물을 만나는지 확인한다.
위 두 가지중 어느 하나라도 해당된다면, 로봇 강아지는 해당 명령을 무시하고 다음 명령을 수행한다.
공원의 가로 길이가 W, 세로 길이가 H라고 할 때, 공원의 좌측 상단의 좌표는 (0, 0), 우측 하단의 좌표는 (H - 1, W - 1) 이다.공원을 나타내는 문자열 배열 park, 로봇 강아지가 수행할 명령이 담긴 문자열 배열 routes가 매개변수로 주어질 때, 로봇 강아지가 모든 명령을 수행 후 놓인 위치를 [세로 방향 좌표, 가로 방향 좌표] 순으로 배열에 담아 반환하도록 solution 함수를 완성해라.
"""
# 2) 제한 사항

"""3 ≤ park[i]의 길이 ≤ 50
park[i]는 다음 문자들로 이루어져 있으며 시작지점은 하나만 주어진다.
S : 시작 지점
O : 이동 가능한 통로
X : 장애물
park는 직사각형 모양이다.
1 ≤ routes의 길이 ≤ 50
routes의 각 원소는 로봇 강아지가 수행할 명령어를 나타낸다.
로봇 강아지는 routes의 첫 번째 원소부터 순서대로 명령을 수행한다.
routes의 원소는 "op n"과 같은 구조로 이루어져 있으며, op는 이동할 방향, n은 이동할 칸의 수를 의미한다.
op는 다음 네 가지중 하나로 이루어져 있다.
N : 북쪽으로 주어진 칸만큼 이동한다.
S : 남쪽으로 주어진 칸만큼 이동한다.
W : 서쪽으로 주어진 칸만큼 이동한다.
E : 동쪽으로 주어진 칸만큼 이동한다.
1 ≤ n ≤ 9"""


# 3) 입출력 예시


"""입출력 예 #1

park가 ["SOO","OOO","OOO"]이고 routes가 ["E 2","S 2","W 1"]이면, 입력된 명령대로 동쪽으로 2칸, 남쪽으로 2칸, 서쪽으로 1칸 이동하면 [0,0] -> [0,2] -> [2,2] -> [2,1]이 된다.
 

입출력 예 #2

park가 ["SOO","OXX","OOO"]이고 routes가 ["E 2","S 2","W 1"]이면 동쪽으로 2칸, 남쪽으로 2칸, 서쪽으로 1칸 이동해야하지만 남쪽으로 2칸 이동할 때 장애물이 있는 칸을 지나기 때문에 해당 명령을 제외한 명령들만 따른다. 결과적으로는 [0,0] -> [0,2] -> [0,1]이 된다.

 

입출력 예 #3

park가 ["OSO","OOO","OXO","OOO"]이고 routes가 ["E 2","S 3","W 1"]이면 처음 입력된 명령은 공원을 나가게 되고 두 번째로 입력된 명령 또한 장애물을 지나가게 되므로 두 입력은 제외한 세 번째 명령만 따르므로 결과는 다음과 같다. [0,1] -> [0,0]"""

# 4) 코드 설명

"""solution 함수에서 park, routes의 값을 입력 받는다.
값을 반환하기 위해 answer이라는 빈 리스트를 만든다.
반복문을 통해 park 리스트의 길이 만큼 i에 값을 1씩 증가시키면서 만약 "S"가 park 리스트의 해당 i값의 위치에 있다면 answer을 [i, park[i].index("S")]로 저장하여 시작점의 위치를 찾고 break로 반복문을 빠져나온다.
그 후 반복문을 통해 j에 routes의 값을 하나씩 집어넣으면 최종 위치를 찾기 위해 반복한다.
반복문 안에서는 direction에 j에 인덱스 0번째 위치에 값을 가져와 방향을 저장한다.
distance에는 j에 인덱스 2번째 위치에 값을 가져와 int 함수를 통해 정수로 변환하여 이동할 거리를 저장한다.
만약 direction == "E"이고 answer[1] 더하기 distance한 값이 park[0]의 길이보다 작고 all함수를 사용하여 park 리스트에 모든 answer[0] 값에 위치의 answer[1]+k위치 값이 "X"가 아니라면 , 즉 이동할 방향이 동쪽이고 현재 x좌표 더하기 이동할 거리를 한 값이 가로 길이 끝점보다 작고 이동할 위치 사이에 장애물이 없다면 answer[1] 값에 distance를 더하여 x좌표를 바꾼다.
만약 direction == "W"이고 answer[1] 뺴기 distance한 값이 0보다 크거나 같고 all함수를 사용하여 park 리스트에 모든 answer[0] 값에 위치의 answer[1]-k 위치 값이 "X"가 아니라면 , 즉 이동할 방향이 서쪽이고 현재 x좌표 빼기 이동할 거리를 한 값이 가로 길이 시작점보다 크거나 같고 이동할 위치 사이에 장애물이 없다면 answer[1] 값에 distance를 빼서 x좌표를 바꾼다.
만약 direction == "S"이고 answer[0] 더하기 distance한 값이 park의 길이보다 작고 all함수를 사용하여 park 리스트에 모든 answer[0]+k한 값에 위치의 answer[1] 위치 값이 "X"가 아니라면 , 즉 이동할 방향이 남쪽이고 현재 y좌표 더하기 이동할 거리를 한 값이 세로 길이 끝점보다 작고 이동할 위치 사이에 장애물이 없다면 answer[0] 값에 distance를 더해서 Y좌표를 바꾼다.
만약 direction == "N"이고 answer[0] 빼기 distance한 값이 0보다 크거나 같고 all함수를 사용하여 park 리스트에 모든 answer[0]-k한 값에 위치의 answer[1] 위치 값이 "X"가 아니라면 , 즉 이동할 방향이 북쪽이고 현재 y좌표 뺴기 이동할 거리를 한 값이 세로 길이 시작점보다 크거나 같고 이동할 위치 사이에 장애물이 없다면 answer[0] 값에 distance를 빼서 Y좌표를 바꾼다.
그 후 반복문이 종료 되면 answer 값을 반환한다."""

# 문제 출처 : 프로그래머스
