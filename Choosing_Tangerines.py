def solution(k, tangerine):
    from collections import Counter

    size_count = Counter(tangerine)
    sorted_counts = sorted(size_count.values(), reverse=True)

    total = 0
    answer = 0

    for count in sorted_counts:
        total += count
        answer += 1
        if total >= k:
            break

    return answer


# 1) 문제 설명

"""경화는 과수원에서 귤을 수확했다. 경화는 수확한 귤 중 'k'개를 골라 상자 하나에 담아 판매하려고 한다. 그런데 수확한 귤의 크기가 일정하지 않아 보기에 좋지 않다고 생각한 경화는 귤을 크기별로 분류했을 때 서로 다른 종류의 수를 최소화하고 싶다.
예를 들어, 경화가 수확한 귤 8개의 크기가 [1, 3, 2, 5, 4, 5, 2, 3] 이라고 합시다. 경화가 귤 6개를 판매하고 싶다면, 크기가 1, 4인 귤을 제외한 여섯 개의 귤을 상자에 담으면, 귤의 크기의 종류가 2, 3, 5로 총 3가지가 되며 이때가 서로 다른 종류가 최소일 때이다
경화가 한 상자에 담으려는 귤의 개수 k와 귤의 크기를 담은 배열 tangerine이 매개변수로 주어진다. 경화가 귤 k개를 고를 때 크기가 서로 다른 종류의 수의 최솟값을 반환하도록 solution 함수를 작성해라."""

# 2) 제한 사항

"""1 ≤ k ≤ tangerine의 길이 ≤ 100,000
1 ≤ tangerine의 원소 ≤ 10,000,000"""


# 3) 입출력 예시

"""입출력 예 #1


k가 4이고 tangerine이 [1, 3, 2, 5, 4, 5, 2, 3]이면, 경화는 크기가 2인 귤 2개와 3인 귤 2개 또는 2인 귤 2개와 5인 귤 2개 또는 3인 귤 2개와 5인 귤 2개로 귤을 판매할 수 있다. 이때의 크기 종류는 2가지로 이 값이 최소가 된다."""

# 4) 코드 설명

"""solution 함수에서 k, tangerine의 값을 입력 받는다.
Counter 모듈을 불러온다.
Counter()을 사용하여 귤의 크기를 세고, 크기별 귤의 개수를 딕셔너리 형태로 저장한다.
sorted 함수를 이용하여 size_count 딕셔너리의 values 값만 내림차순형태로 정렬하여 sorted_counts에 리스트 형태로 저장한다.
현재까지 선택한 서로 다른 귤의 종류 수를 구하기 위해 answer에 초기 값 0으로 설정한다.
현재까지 담은 귤의 수를 구하기 위해 total에 초기 값 0으로 설정한다.
반복문을 통해 sorted_counts 리스트의 요소 값을 하나씩 count에 넣어준다.
현재 담긴 귤의 개수를 구하기 위해 total에 count값을 더해준다.
귤을 담고 나서 answer의 값을 1 증가시켜 현재 몇 종류의 귤이 담겼는지 구한다.
만약 total이 k보다 크거나 같으면 원하는 만큼 담긴 것이므로 break를 사용하여 반복문을 종료한다.
그 후 answer 값을 반환한다."""

# 문제 출처 : 프로그래머스
