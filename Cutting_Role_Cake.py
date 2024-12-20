from collections import Counter  # 각 토핑의 종류를 세기 위한 Counter 모듈

def solution(topping):
    answer = 0  # 공평하게 나눌 수 있는 경우의 수를 세기 위한 변수
    left_counter = Counter()  # 왼쪽 조각의 토핑 종류와 개수를 저장하는 변수
    right_counter = Counter(topping)  # 오른쪽 조각의 토핑 종류와 개수를 저장하는 변수

    # topping을 순회하며 왼쪽과 오른쪽 조각의 토핑 상태를 업데이트
    for i in range(len(topping) - 1): 
        
        left_counter[topping[i]] += 1  # 현재 토핑을 왼쪽 조각에 추가
        right_counter[topping[i]] -= 1  # 현재 토핑을 오른쪽 조각에서 제거
        
        # 만약 현재 토핑의 개수가 0이 되면, right_counter에서 해당 토핑을 삭제
        if right_counter[topping[i]] == 0:
            del right_counter[topping[i]]

        # 왼쪽 조각과 오른쪽 조각의 고유 토핑 수가 같으면 경우의 수를 증가
        if len(left_counter) == len(right_counter):
            answer += 1 

    return answer 




#1) 문제 설명

'''철수는 롤케이크를 두 조각으로 잘라서 동생과 한 조각씩 나눠 먹으려고 한다. 이 롤케이크에는 여러가지 토핑들이 일렬로 올려져 있다. 철수와 동생은 롤케이크를 공평하게 나눠먹으려 하는데, 그들은 롤케이크의 크기보다 롤케이크 위에 올려진 토핑들의 종류에 더 관심이 많다. 그래서 잘린 조각들의 크기와 올려진 토핑의 개수에 상관없이 각 조각에 동일한 가짓수의 토핑이 올라가면 공평하게 롤케이크가 나누어진 것으로 생각한다.
예를 들어, 롤케이크에 4가지 종류의 토핑이 올려져 있다고 하자. 토핑들을 1, 2, 3, 4와 같이 번호로 표시했을 때, 케이크 위에 토핑들이 [1, 2, 1, 3, 1, 4, 1, 2] 순서로 올려져 있다. 만약 세 번째 토핑(1)과 네 번째 토핑(3) 사이를 자르면 롤케이크의 토핑은 [1, 2, 1], [3, 1, 4, 1, 2]로 나뉘게 된다. 철수가 [1, 2, 1]이 놓인 조각을, 동생이 [3, 1, 4, 1, 2]가 놓인 조각을 먹게 되면 철수는 두 가지 토핑(1, 2)을 맛볼 수 있지만, 동생은 네 가지 토핑(1, 2, 3, 4)을 맛볼 수 있으므로, 이는 공평하게 나누어진 것이 아니다. 만약 롤케이크의 네 번째 토핑(3)과 다섯 번째 토핑(1) 사이를 자르면 [1, 2, 1, 3], [1, 4, 1, 2]로 나뉘게 된다. 이 경우 철수는 세 가지 토핑(1, 2, 3)을, 동생도 세 가지 토핑(1, 2, 4)을 맛볼 수 있으므로, 이는 공평하게 나누어진 것이다. 공평하게 롤케이크를 자르는 방법은 여러가지 일 수 있다. 위의 롤케이크를 [1, 2, 1, 3, 1], [4, 1, 2]으로 잘라도 공평하게 나뉜다. 어떤 경우에는 롤케이크를 공평하게 나누지 못할 수도 있다.
롤케이크에 올려진 토핑들의 번호를 저장한 정수 배열 topping이 매개변수로 주어질 때, 롤케이크를 공평하게 자르는 방법의 수를 반환하도록 solution 함수를 완성해라.'''

#2) 제한 사항

'''1 ≤ topping의 길이 ≤ 1,000,000
1 ≤ topping의 원소 ≤ 10,000'''

#3) 입출력 예시 


'''입출력 예 #1



topping이 [1, 2, 1, 3, 1, 4, 1, 2]이면, 롤케이크를 [1, 2, 1, 3], [1, 4, 1, 2] 또는 [1, 2, 1, 3, 1], [4, 1, 2]와 같이 자르면 철수와 동생은 각각 세 가지 토핑을 맛볼 수 있다. 이 경우 공평하게 롤케이크를 나누는 방법은 위의 두 가지만 존재한다.



입출력 예 #2



topping이 [1, 2, 3, 1, 4]이면,롤케이크를 공평하게 나눌 수 없다.'''

#4) 코드 설명

'''각 토핑의 종류를 세기위해 Counter이라는 모듈을 불러온다.
solution 함수에 topping 값을 입력 받는다.
공평하게 나눌 수 있는 경우의 수를 세기 위하여 answer을 0으로 초기화한다.
왼쪽 조각의 토핑 종류와 개수를 세기 위해 left_counter에 Counter() 함수를 사용하여 저장한다.
오른쪽 조각의 토핑 종류와 개수를 세기 위해 right_count에 Counter() 함수를 이용하여 초기에는 자르기 전이므로 topping 리스트에서 종류와 개수를 구해서 저장한다.
반복문을 통해 0부터 topping의 길이 빼기 2까지의 값을 하나씩 i에 집어넣에 topping을 순회하며 왼쪽과 오른쪽 조각의 토핑 상태를 업데이트 한다.
반복문 안에서는 left_counter에 topping[i]를 1 추가하여, 현재 토핑을 왼쪽 조각에 추가한다.
그 후 rigt_counter에 topping[i]를 1 빼서, 현재 토핑을 오른쪽 조각에서 하나 제거한다.
만약 right_counter[topping[i]]의 값이 0이라면, 즉 오른쪽에 현재 토핑의 개수가 0이라면 del 함수를 사용하여 해당 토핑을 아예 삭제해서 오른쪽에는 해당 토핑이 없는 것으로 처리한다.
만약 left_counter의 길이와 right_counter의 길이가 같다면, 즉 왼쪽과 오른쪽의 토핑의 개수가 같다면 answer의 값을 1 증가 시켜 경우의 수를 추가한다.
모든 코드가 끝나면 answer의 값을 반환한다.'''

#문제 출처 : 프로그래머스