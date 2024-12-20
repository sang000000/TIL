def solution(food):
    answer = "0"
    for i in range(len(food) - 1, 0, -1):
        if food[i] > 1:
            if food[i] % 2 != 0:
                answer = answer.center(len(answer) + food[i] - 1, str(i))
            elif food[i] % 2 == 0:
                answer = answer.center(len(answer) + food[i], str(i))
    return answer


# 1) 문제 설명

"""수웅이는 매달 주어진 음식을 빨리 먹는 푸드 파이트 대회를 개최합니다. 이 대회에서 선수들은 1대 1로 대결하며, 매 대결마다 음식의 종류와 양이 바뀐다. 대결은 준비된 음식들을 일렬로 배치한 뒤, 한 선수는 제일 왼쪽에 있는 음식부터 오른쪽으로, 다른 선수는 제일 오른쪽에 있는 음식부터 왼쪽으로 순서대로 먹는 방식으로 진행된다. 중앙에는 물을 배치하고, 물을 먼저 먹는 선수가 승리하게 된다.예를 들어, 3가지의 음식이 준비되어 있으며, 칼로리가 적은 순서대로 1번 음식을 3개, 2번 음식을 4개, 3번 음식을 6개 준비했으며, 물을 편의상 0번 음식이라고 칭한다면, 두 선수는 1번 음식 1개, 2번 음식 2개, 3번 음식 3개씩을 먹게 되므로 음식의 배치는 "1223330333221"이 된다. 따라서 1번 음식 1개는 대회에 사용하지 못한다.
수웅이가 준비한 음식의 양을 칼로리가 적은 순서대로 나타내는 정수 배열 food가 주어졌을 때, 대회를 위한 음식의 배치를 나타내는 문자열을 return 하는 solution 함수를 완성해라.
이때, 대회의 공정성을 위해 두 선수가 먹는 음식의 종류와 양이 같아야 하며, 음식을 먹는 순서도 같아야 한다. 또한, 이번 대회부터는 칼로리가 낮은 음식을 먼저 먹을 수 있게 배치하여 선수들이 음식을 더 잘 먹을 수 있게 하려고 한다. 이번 대회를 위해 수웅이는 음식을 주문했는데, 대회의 조건을 고려하지 않고 음식을 주문하여 몇 개의 음식은 대회에 사용하지 못하게 되었다."""

# 2) 제한 사항

"""2 ≤ food의 길이 ≤ 9
1 ≤ food의 각 원소 ≤ 1,000
food에는 칼로리가 적은 순서대로 음식의 양이 담겨 있다.
food[i]는 i번 음식의 수이다.
food[0]은 수웅이가 준비한 물의 양이며, 항상 1이다.
정답의 길이가 3 이상인 경우만 입력으로 주어진다."""

# 3) 입출력 예시

"""입출력 예 #1

"food"가 [1, 3, 4, 6]이면 두 선수는 1번 음식 1개, 2번 음식 2개, 3번 음식 3개를 먹게 되므로 음식의 배치는 "1223330333221"이다.
입출력 예 #2

"food"가 [1, 7, 1, 2]이면 두 선수는 1번 음식 3개, 3번 음식 1개를 먹게 되므로 음식의 배치는 "111303111"이다."""

# 4) 코드 설명

"""solution 함수에서 food의 값을 입력 받는다.
물은 항상 누군가 마시게 되므로 answer은 '0'이라는 문자열을 만든다.
반복문을 사용하여 food의 길이부터 1까지 1씩 줄이면서 i의 값을 집어넣는다.
반복문 안에서는 만약 food의 요소가 1보다 크면, 즉 음식이 2개 이상일 경우 food의 i번째 음식의 개수가 홀수일 경우에는 두 사람 모두 똑같이 먹어야 하므로 현재 answer의 길이 더하기 i번째 요소 값 뺴기1 한 길이 만큼 center 함수를 이용하여 i번째 음식 나누어준다.
만약 i번째 음식이 짝수이면 마찬가지로 center 함수를 이용하여 현재 answer의 길이 더하기 food 리스트의 i번째 요소 값을 더한 만큼 i번째 음식을 가운데 정렬을 하여 똑같이 나누어준다.
반복문이 끝나면 answer을 반환한다."""

# 문제 출처 : 프로그래머스
