def solution(order):
    # 스택과 컨테이너 벨트 초기화
    stack = []
    container = 1
    answer = 0

    # order를 순회하며 상자를 실을 수 있는지 확인
    for target in order:
        # 컨테이너 벨트에서 상자를 가져와야 하는 경우
        while container <= target:
            stack.append(container)
            container += 1
        
        # 스택에서 상자를 실을 수 있는 경우
        if stack and stack[-1] == target:
            stack.pop()
            answer += 1
        else:
            # 원하는 순서대로 상자를 실을 수 없는 경우
            break

    return answer



#1) 문제 설명

'''영재는 택배상자를 트럭에 싣는 일을 한다. 영재가 실어야 하는 택배상자는 크기가 모두 같으며 1번 상자부터 n번 상자까지 번호가 증가하는 순서대로 컨테이너 벨트에 일렬로 놓여 영재에게 전달된다. 컨테이너 벨트는 한 방향으로만 진행이 가능해서 벨트에 놓인 순서대로(1번 상자부터) 상자를 내릴 수 있다. 하지만 컨테이너 벨트에 놓인 순서대로 택배상자를 내려 바로 트럭에 싣게 되면 택배 기사님이 배달하는 순서와 택배상자가 실려 있는 순서가 맞지 않아 배달에 차질이 생긴다. 따라서 택배 기사님이 미리 알려준 순서에 맞게 영재가 택배상자를 실어야 한다.
만약 컨테이너 벨트의 맨 앞에 놓인 상자가 현재 트럭에 실어야 하는 순서가 아니라면 그 상자를 트럭에 실을 순서가 될 때까지 잠시 다른 곳에 보관해야 한다. 하지만 고객의 물건을 함부로 땅에 둘 수 없어 보조 컨테이너 벨트를 추가로 설치하였다. 보조 컨테이너 벨트는 앞 뒤로 이동이 가능하지만 입구 외에 다른 면이 막혀 있어서 맨 앞의 상자만 뺄 수 있다(즉, 가장 마지막에 보조 컨테이너 벨트에 보관한 상자부터 꺼내게 된다). 보조 컨테이너 벨트를 이용해도 기사님이 원하는 순서대로 상자를 싣지 못 한다면, 더 이상 상자를 싣지 않는다.
예를 들어, 영재가 5개의 상자를 실어야 하며, 택배 기사님이 알려준 순서가 기존의 컨테이너 벨트에 네 번째, 세 번째, 첫 번째, 두 번째, 다섯 번째 놓인 택배상자 순서인 경우, 영재는 우선 첫 번째, 두 번째, 세 번째 상자를 보조 컨테이너 벨트에 보관한다. 그 후 네 번째 상자를 트럭에 싣고 보조 컨테이너 벨트에서 세 번째 상자 빼서 트럭에싣는다. 다음으로 첫 번째 상자를 실어야 하지만 보조 컨테이너 벨트에서는 두 번째 상자를, 기존의 컨테이너 벨트에는 다섯 번째 상자를 꺼낼 수 있기 때문에 더이상의 상자는 실을 수 없다. 따라서 트럭에는 2개의 상자만 실리게 된다.
택배 기사님이 원하는 상자 순서를 나타내는 정수 배열 order가 주어졌을 때, 영재가 몇 개의 상자를 실을 수 있는지 반환하는 solution 함수를 완성하라.'''


#2) 제한 사항

'''1 ≤ order의 길이 ≤ 1,000,000
order는 1이상 order의 길이 이하의 모든 정수가 한번씩 등장한다.
order[i]는 기존의 컨테이너 벨트에 order[i]번째 상자를 i+1번째로 트럭에 실어야 함을 의미한다.'''

#3) 입출력 예시 

'''입출력 예 #1

order가 [5, 4, 3, 2, 1]이면, 5가 반환된다.'''

#4) 코드 설명

'''solution 함수에 order 값을 입력 받는다.
가장 최근에 넣은 상자를 꺼내기 위해  stack이라는 빈 리스트를 만든다.
컨테이너 벨트에서 현재 상자의 번호를 알기 위해 container를 1로 설정한다.
실을 수 있는 상자 개수를 세기 위해 answer 값을 0으로 초기화한다.
for문을 통해 order 리스트의 요소 값을 하나씩 target에 넣는다.
for문 안에서는 while문을 실행하여 container가 target보다 작거나 같을 동안 반복한다.
while문 안에서는 container 값이 target보다 작거나 같다는 의미이므로 조건을 만족하는 container의 값을 stack에 추가하여 stack에 보관하고 container의 값을 1 증가 시킨다.
while문이 종료되면 target보다 작은 번호들의 상자는 모두 stack에 보관되게 된다.
만약 stack이 비어 있지 않고 stack의 마지막 요소 값이 target과 같다면, pop 함수를 사용하여 마지막 요소 값을 삭제하고 answer의 값을 1 증가 시킨다.
그렇지 않다면 for문을 종료한다.
for문이 종료되면 answer의 값을 반환한다.'''

#문제 출처 : 프로그래머스