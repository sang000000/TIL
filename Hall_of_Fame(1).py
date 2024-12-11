def solution(k, score):
    answer = []
    hall = []
    for i in score:
        hall.append(i)
        hall.sort()
        if len(hall) > k:
            del hall[0]
        answer.append(hall[0])
    return answer


# 1) 문제 설명

""""명예의 전당"이라는 TV 프로그램에서는 매일 1명의 가수가 노래를 부르고, 시청자들의 문자 투표수로 가수에게 점수를 부여한다. 매일 출연한 가수의 점수가 지금까지 출연 가수들의 점수 중 상위 k번째 이내이면 해당 가수의 점수를 명예의 전당이라는 목록에 올려 기념한다. 즉 프로그램 시작 이후 초기에 k일까지는 모든 출연 가수의 점수가 명예의 전당에 오르게 된다. k일 다음부터는 출연 가수의 점수가 기존의 명예의 전당 목록의 k번째 순위의 가수 점수보다 더 높으면, 출연 가수의 점수가 명예의 전당에 오르게 되고 기존의 k번째 순위의 점수는 명예의 전당에서 내려오게 된다.
명예의 전당 목록의 점수의 개수 k, 1일부터 마지막 날까지 출연한 가수들의 점수인 score가 주어졌을 때, 매일 발표된 명예의 전당의 최하위 점수를 return하는 solution 함수를 완성해라.
이 프로그램에서는 매일 "명예의 전당"의 최하위 점수를 발표한다. 예를 들어, k = 3이고, 7일 동안 진행된 가수의 점수가 [10, 100, 20, 150, 1, 100, 200]이라면, 명예의 전당에서 발표된 점수는 아래의 그림과 같이 [10, 10, 10, 20, 20, 100, 100]이다."""

# 2) 제한 사항

"""2 ≤ food의 길이 ≤ 9
1 ≤ food의 각 원소 ≤ 1,000
food에는 칼로리가 적은 순서대로 음식의 양이 담겨 있다.
food[i]는 i번 음식의 수이다.
food[0]은 수웅이가 준비한 물의 양이며, 항상 1이다.
정답의 길이가 3 이상인 경우만 입력으로 주어진다."""

# 3) 입출력 예시

"""입출력 예 #1

k가 4이고 score가 [0, 300, 40, 300, 20, 70, 150, 50, 500, 1000]이면,
[0, 0, 0, 0, 20, 40, 70, 70, 150, 300]을 return한다."""

# 4) 코드 설명

"""solution 함수에서 k와 score의 값을 입력 받는다.
발표 점수를 저장하기 위해 answer이라는 빈 리스트를 만든다.
명예의 전당에 들어왔는지 확인하기 위해 hall이라는 빈 리스트를 만든다.
score에 점수를 하나씩 i에 집어 넣으며 append 함수를 이용해 hall 리스트에 추가한다.
그 후 sort 함수를 이용해 점수가 낮은 순으로 정렬한다.
만약 score를 추가하고 나서 hall의 길이가 k보다 크다면 명예의 전당의 수용 인원을 넘은 것이므로 젤 작은 값인 맨 앞 숫자를 삭제한다.
그 후 answer 리스트에 최하위 점수인 맨 앞 숫자를 집어넣는다.
반복문이 끝나면 그 값을 반환한다."""

# 문제 출처 : 프로그래머스
