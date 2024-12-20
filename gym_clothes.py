def solution(n, lost, reserve):
    # 여벌 체육복을 가진 학생 중 도난당한 학생 제거
    lost_set = set(lost) - set(reserve)
    reserve_set = set(reserve) - set(lost)

    # 초기 학생 수 (도난당한 학생 제외)
    answer = n - len(lost_set)

    # 여벌 체육복을 가진 학생을 정렬하여 처리
    for r in sorted(reserve_set):
        if r - 1 in lost_set:
            lost_set.remove(r - 1)  # 앞 학생에게 빌려줌
            answer += 1
        elif r + 1 in lost_set:
            lost_set.remove(r + 1)  # 뒤 학생에게 빌려줌
            answer += 1

    return answer


# 1) 문제 설명

"""점심시간에 도둑이 들어, 일부 학생이 체육복을 도난당했다. 다행히 여벌 체육복이 있는 학생이 이들에게 체육복을 빌려주려 한다. 학생들의 번호는 체격 순으로 매겨져 있어, 바로 앞번호의 학생이나 바로 뒷번호의 학생에게만 체육복을 빌려줄 수 있다. 예를 들어, 4번 학생은 3번 학생이나 5번 학생에게만 체육복을 빌려줄 수 있다. 체육복이 없으면 수업을 들을 수 없기 때문에 체육복을 적절히 빌려 최대한 많은 학생이 체육수업을 들어야 한다.
전체 학생의 수 n, 체육복을 도난당한 학생들의 번호가 담긴 배열 lost, 여벌의 체육복을 가져온 학생들의 번호가 담긴 배열 reserve가 매개변수로 주어질 때, 체육수업을 들을 수 있는 학생의 최댓값을 return 하도록 solution 함수를 작성해라."""

# 2) 제한 사항

"""전체 학생의 수는 2명 이상 30명 이하이다.
체육복을 도난당한 학생의 수는 1명 이상 n명 이하이고 중복되는 번호는 없다.
여벌의 체육복을 가져온 학생의 수는 1명 이상 n명 이하이고 중복되는 번호는 없다.
여벌 체육복이 있는 학생만 다른 학생에게 체육복을 빌려줄 수 있다.
여벌 체육복을 가져온 학생이 체육복을 도난당했을 수 있다. 이때 이 학생은 체육복을 하나만 도난당했다고 가정하며, 남은 체육복이 하나이기에 다른 학생에게는 체육복을 빌려줄 수 없다."""

# 3) 입출력 예시


"""입출력 예 #1

n이 5이고 lost가 [2,4]이고 reserve가 [1,3,5]이면, 1번 학생이 2번 학생에게 체육복을 빌려주고, 3번 학생이나 5번 학생이 4번 학생에게 체육복을 빌려주면 학생 5명이 체육수업을 들을 수 있으므로 5를 반환한다.


입출력 예 #2

n이 5이고 lost가 [2,4]이고 reserve가 [3]이면, 3번 학생이 2번 학생이나 4번 학생에게 체육복을 빌려주면 학생 4명이 체육수업을 들을 수 있으므로 4를 반환한다.
입출력 예 #3

n이 3이고 lost가 [3]이고 reserve가 [1]이면, 제한 사항에 의해 1번 학생은 3번 학생에게 체육복을 빌려줄 수 없으므로 학생 2명이 체육수업을 들을 수 있으므로 2를 반환한다."""

# 4) 코드 설명
"""solution 함수에서 n, lost, reserve의 값을 입력 받는다.
set 함수를 사용하여 도난 당한 사람 중 여벌 체육복을 갖고 있는 사람을 찾아 제거하여 lost_set에 저장한다.
set 함수를 이용하여 여벌 체육복을 갖고 있는 사람 중 도난 당한 사람을 제거하여 reserve_set에 저장한다.
초기 학생의 수는 전체 학생 수(n)에서 도난당한 학생의 수(lost_set의 길이)를 뻬사 answer에 저장한다.
반복문을 통해 reserve_set을 오름차순으로 정렬한 후 하나씩 r에 집어넣는다.
반복문 안에서는 만약 r 뺴기 1한 값이 lost_set에 있다면, lost-set에서 r-1한 값을 제거하고 answer에 값을 1증가 시킨다. 즉 여벌 체육복을 갖고 있는 사람 중 앞 학생이 도난당한 학생이라면, 앞 학생에게 빌려준다는 의미이다.
위에 조건을 만족하지 못했다면 만약 r 더하기 1한 값이 lost_set에 있다면, lost-set에서 r+1한 값을 제거하고 answer에 값을 1증가 시킨다. 즉 여벌 체육복을 갖고 있는 사람 중 뒤 학생이 도난당한 학생이라면, 뒷 학생에게 빌려준다는 의미이다.
반복문이 끝나면 answer 값을 반환한다."""

# 문제 출처 : 프로그래머스
