def solution(cards1, cards2, goal):
    for i in range(len(goal)):
        if len(cards1) > 0 and goal[i] == cards1[0]:
            del cards1[0]
        elif len(cards2) > 0 and goal[i] == cards2[0]:
            del cards2[0]
        else:
            answer = "No"
            break
        answer = "Yes"
    return answer


# 1) 문제 설명

"""코니는 영어 단어가 적힌 카드 뭉치 두 개를 선물로 받았다. 코니는 다음과 같은 규칙으로 카드에 적힌 단어들을 사용해 원하는 순서의 단어 배열을 만들 수 있는지 알고 싶다.예를 들어 첫 번째 카드 뭉치에 순서대로 ["i", "drink", "water"], 두 번째 카드 뭉치에 순서대로 ["want", "to"]가 적혀있을 때 ["i", "want", "to", "drink", "water"] 순서의 단어 배열을 만들려고 한다면 첫 번째 카드 뭉치에서 "i"를 사용한 후 두 번째 카드 뭉치에서 "want"와 "to"를 사용하고 첫 번째 카드뭉치에 "drink"와 "water"를 차례대로 사용하면 원하는 순서의 단어 배열을 만들 수 있다.
원하는 카드 뭉치에서 카드를 순서대로 한 장씩 사용한다.
한 번 사용한 카드는 다시 사용할 수 없다.
카드를 사용하지 않고 다음 카드로 넘어갈 수 없다.
기존에 주어진 카드 뭉치의 단어 순서는 바꿀 수 없다.
예를 들어 첫 번째 카드 뭉치에 순서대로 ["i", "drink", "water"], 두 번째 카드 뭉치에 순서대로 ["want", "to"]가 적혀있을 때 ["i", "want", "to", "drink", "water"] 순서의 단어 배열을 만들려고 한다면 첫 번째 카드 뭉치에서 "i"를 사용한 후 두 번째 카드 뭉치에서 "want"와 "to"를 사용하고 첫 번째 카드뭉치에 "drink"와 "water"를 차례대로 사용하면 원하는 순서의 단어 배열을 만들 수 있다.
문자열로 이루어진 배열 cards1, cards2와 원하는 단어 배열 goal이 매개변수로 주어질 때, cards1과 cards2에 적힌 단어들로 goal를 만들 있다면 "Yes"를, 만들 수 없다면 "No"를 return하는 solution 함수를 완성해라."""

# 2) 제한 사항

"""1 ≤ cards1의 길이, cards2의 길이 ≤ 10
1 ≤ cards1[i]의 길이, cards2[i]의 길이 ≤ 10
cards1과 cards2에는 서로 다른 단어만 존재한다.

2 ≤ goal의 길이 ≤ cards1의 길이 + cards2의 길이
1 ≤ goal[i]의 길이 ≤ 10
goal의 원소는 cards1과 cards2의 원소들로만 이루어져 있다.
cards1, cards2, goal의 문자열들은 모두 알파벳 소문자로만 이루어져 있다."""

# 3) 입출력 예시

"""입출력 예 #1

cards1가 ["i", "water", "drink"]이고 cards2가  ["want", "to"]이면, cards1에서 "i"를 사용하고 cards2에서 "want"와 "to"를 사용하여 "i want to"까지는 만들 수 있지만 "water"가 "drink"보다 먼저 사용되어야 하기 때문에 해당 문장을 완성시킬 수 없다. 따라서 "No"를 반환한다."""

# 4) 코드 설명

"""solution 함수에서 cards1과 cards2와 goal의 값을 입력 받는다.
goal과 비교하기 위해 반복문을 이용하여 goal 리스트의 길이만큼 i값을 1씩 증가하며 요소 값을 비교한다.
만약 cards1의 길이가 1이상이고 goal의 i 번째 요소가 cards1의 첫번째 위치에 있다면 cards1의 카드를 사용한거로 생각하고 그 요소 값을 지운다.
만약 cards2의 길이가 1이상이고 goal의 i 번째 요소가 cards2의 첫번째 위치에 있다면 cards2의 카드를 사용한거로 생각하고 그 요소 값을 지운다.
만약 둘다 아니라면 현재 goal 리스트의 i번째 요소 값의 카드가 나올 차례인데 cards1과 cards2 모두 순서상 사용할 수 없다는 뜻이므로 문장을 만들수 없어 answer 값을 "No"로 하고 반복문을 즉시 종료한다.
위 조건문이 발동이 되지 않았다면 문장을 완성할 수 있다는 의미이므로 answer 값은 "Yes"가 된다.
반복문이 끝났으면 answer 값을 반환한다."""

# 문제 출처 : 프로그래머스
