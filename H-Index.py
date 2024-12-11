def solution(citations):
    answer = 0
    count = 0
    for i in range(len(citations) + 1):
        if i != 0:
            count = len(list(filter(lambda x: x >= i, citations)))
        if i <= count:
            answer = i
        elif i > count:
            break

    return answer


# 1) 문제 설명

"""H-Index는 과학자의 생산성과 영향력을 나타내는 지표이다. 어느 과학자의 H-Index를 나타내는 값인 h를 구하려고 한다. H-Index는 다음과 같이 구한다.
어떤 과학자가 발표한 논문 n편 중, h번 이상 인용된 논문이 h편 이상이고 나머지 논문이 h번 이하 인용되었다면 h의 최댓값이 이 과학자의 H-Index이다.
어떤 과학자가 발표한 논문의 인용 횟수를 담은 배열 citations가 매개변수로 주어질 때, 이 과학자의 H-Index를 반환하도록 solution 함수를 작성해라."""

# 2) 제한 사항

"""과학자가 발표한 논문의 수는 1편 이상 1,000편 이하이다.
논문별 인용 횟수는 0회 이상 10,000회 이하이다.
"""

# 3) 입출력 예시


"""입출력 예 #1



citations가 [3, 0, 6, 1, 5]이면, 이 과학자가 발표한 논문의 수는 5편이고, 그중 3편의 논문은 3회 이상 인용되었다. 그리고 나머지 2편의 논문은 3회 이하 인용되었기 때문에 이 과학자의 H-Index는 3이다."""

# 4) 코드 설명

"""solution 함수에서 citations의 값을 입력 받는다.
함수를 여러번 사용하기 위해 answer과 count를 0으로 초기화 해준다.
반복문에서 i에 0부터 citations의 길이까지 값을 하나씩 할당해준다.
반복문 안에서는 만약 i가 0이 아니라면 lambda 함수와 filter 함수를 사용하여 citations의 각 요소들이 i보다 큰지 비교하여 크면 그 값들만 리스트로 만들고 len 함수를 이용해 개수를 세 count에 저장한다.
만약 i가 count보다 작거나 같으면 answer에 i 값을 할당하여 현재 H-index 값으로 설정한다.
그렇지 않고 만약 i가 count보다 크다면,  예를 들어 i는 5이고 count는 3이라면 i는 논문이 인용된 수의 기준점이고 count는 i 이상 논문이 인용된 수이므로 i가 더 크다는 것은 H-index 좋 건에 맞지 않는다는 뜻이다. 그래서 break를 통해 반복문을 종료해준다.
반복문이 끝났다면 answer에는 H-index에 최대값이 저장되어 있으므로 그 값을 반환한다. """

# 문제 출처 : 프로그래머스
