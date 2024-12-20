from collections import deque


def solution(priorities, location):
    # 프로세스와 우선순위를 함께 저장하는 큐 초기화
    queue = deque([(idx, priority) for idx, priority in enumerate(priorities)])
    count = 0  # 실행된 프로세스 카운트

    while queue:
        current = queue.popleft()  # 큐에서 프로세스를 꺼냄
        # 현재 프로세스보다 우선순위가 높은 프로세스가 있는지 확인
        if any(current[1] < other[1] for other in queue):
            queue.append(current)  # 더 높은 우선순위가 있으면 다시 큐에 넣음
        else:
            count += 1  # 프로세스를 실행
            if current[0] == location:  # 만약 실행한 프로세스가 원하는 프로세스라면
                return count  # 몇 번째로 실행되었는지 반환


# 1) 문제 설명

"""운영체제의 역할 중 하나는 컴퓨터 시스템의 자원을 효율적으로 관리하는 것이다. 이 문제에서는 운영체제가 다음 규칙에 따라 프로세스를 관리할 경우 특정 프로세스가 몇 번째로 실행되는지 알아내면 된다.예를 들어 프로세스 4개 [A, B, C, D]가 순서대로 실행 대기 큐에 들어있고, 우선순위가 [2, 1, 3, 2]라면 [C, D, A, B] 순으로 실행하게 된다.
1. 실행 대기 큐(Queue)에서 대기중인 프로세스 하나를 꺼낸다.
2. 큐에 대기중인 프로세스 중 우선순위가 더 높은 프로세스가 있다면 방금 꺼낸 프로세스를 다시 큐에 넣는다.
3. 만약 그런 프로세스가 없다면 방금 꺼낸 프로세스를 실행한다.
  3.1 한 번 실행한 프로세스는 다시 큐에 넣지 않고 그대로 종료된다.
현재 실행 대기 큐(Queue)에 있는 프로세스의 중요도가 순서대로 담긴 배열 priorities와, 몇 번째로 실행되는지 알고싶은 프로세스의 위치를 알려주는 location이 매개변수로 주어질 때, 해당 프로세스가 몇 번째로 실행되는지 반환하도록 solution 함수를 작성해라.
"""
# 2) 제한 사항

"""priorities의 길이는 1 이상 100 이하이다.
priorities의 원소는 1 이상 9 이하의 정수이다.
priorities의 원소는 우선순위를 나타내며 숫자가 클 수록 우선순위가 높습니다.
location은 0 이상 (대기 큐에 있는 프로세스 수 - 1) 이하의 값을 가진다.
priorities의 가장 앞에 있으면 0, 두 번째에 있으면 1 … 과 같이 표현한다."""

# 3) 입출력 예시

"""입출력 예 #1



priorities가 [1, 1, 9, 1, 1, 1]이고 location 0이면, 6개의 프로세스 [A, B, C, D, E, F]가 대기 큐에 있고 중요도가 [1, 1, 9, 1, 1, 1] 이므로 [C, D, E, F, A, B] 순으로 실행된다. 따라서 A는 5번째로 실행된다."""

# 4) 코드 설명

"""양쪽 끝에서 빠르게 추가 및 제거할 수 있는 큐를 생성하기 위해 collections 모듈에서 deque 클래스를 가져온다.
solution 함수에서 priorities, location의 값을 입력 받는다.
반복문을 사용하여 프로세스와 우선 순위를 튜플 형태로 만들어 리스트로 만들고 deque 형태로 변형하여 queue에 저장한다.
순서를 세기 위해 count는 0으로 설정한다.
while문을 queue가 비어 있지 않을 때 동안 실행한다.
while문 안에서는 popleft()를 사용하여 맨 왼쪽에 프로세스를 꺼내 current에 저장한다.
만약 현재 꺼낸 프로세스의 우선 순위가 남아 있는 다른 프로세스의 우선순위보다 낮은 경우가 하나라도 있다면, append 함수를 사용하여 다시 current를 추가하여 다시 큐에 넣는다.
그렇지 않다면 가장 우선 순위가 높다는 뜻이므로 count의 값을 1 증가시키고 만약 실행한 프로세스의 인덱스가 구하고자 하는 location의 값과 같다면 count 값을 반환한다."""

# 문제 출처 : 프로그래머스
