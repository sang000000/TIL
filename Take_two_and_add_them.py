def solution(numbers):
    answer = []
    for i in range(len(numbers) - 1):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] not in answer:
                answer.append(numbers[i] + numbers[j])
    answer.sort()
    return answer


# 1) 문제 설명

"정수 배열 numbers가 주어진다. numbers에서 서로 다른 인덱스에 있는 두 개의 수를 뽑아 더해서 만들 수 있는 모든 수를 배열에 오름차순으로 담아 return 하도록 solution 함수를 완성해라."

# 2) 제한 사항

"""numbers의 길이는 2 이상 100 이하이다.
numbers의 모든 수는 0 이상 100 이하이다."""

# 3) 입출력 예시

"""입출력 예 #1

numbers가 [2,1,3,4,1]이면,
2 = 1 + 1 이다. (1이 numbers에 두 개 있다.)
3 = 2 + 1 이다.
4 = 1 + 3 이다.
5 = 1 + 4 = 2 + 3 이다.
6 = 2 + 4 이다.
7 = 3 + 4 이다.
따라서 [2,3,4,5,6,7] 을 return 해야 한다.

입출력 예 #2

numbers가 [5,0,2,7]이면,
2 = 0 + 2 이다.
5 = 5 + 0 이다.
7 = 0 + 7 = 5 + 2 이다.
9 = 2 + 7 이다.
12 = 5 + 7 이다.
따라서 [2,5,7,9,12] 를 return 해야 한다."""

# 4) 코드 설명

"""solution 함수에서 numbers의 값을 입력 받는다.
값을 저장하기 위한 answer이라는 빈 리스트를 만든다.
더 했을 때 모든 경우의 수를 다 구하기 위해 이중 반복문을 이용하여 numbers의 길이 빼기 1까지 i에 값을 하나씩 증가하며 집어넣고 그 i의 값 더하기 1부터 numbers의 길이 끝까지 j에 값을 하나씩 증가하며 집어 넣는다,
만약 그 i와 j 값들을 이용해 인덱스를 활용하여 numbers 리스트 요소의 값들을 더했을 때 값이 answer 리스트에 없으면 추가해준다.
반복문이 끝나면 sort 함수를 이용해 오름차순으로 정리해주고 반환한다."""

# 문제 출처 : 프로그래머스
