def solution(n, left, right):
    result = []

    # left부터 right까지의 인덱스에 대한 결과를 생성
    for i in range(left, right + 1):
        # (i // n)은 해당 행을 찾고, (i % n)은 해당 열을 찾는다.
        row = i // n
        col = i % n
        # 행과 열 중 더 큰 값을 결과에 추가
        result.append(max(row + 1, col + 1))

    return result


# 1) 문제 설명

"""정수 n, left, right가 주어진다. 다음 과정을 거쳐서 1차원 배열을 만들고자 한다.
n행 n열 크기의 비어있는 2차원 배열을 만든다.
i = 1, 2, 3, ..., n에 대해서, 다음 과정을 반복한다
1행 1열부터 i행 i열까지의 영역 내의 모든 빈 칸을 숫자 i로 채운다.
1행, 2행, ..., n행을 잘라내어 모두 이어붙인 새로운 1차원 배열을 만든다.
새로운 1차원 배열을 arr이라 할 때, arr[left], arr[left+1], ..., arr[right]만 남기고 나머지는 지운다.
정수 n, left, right가 매개변수로 주어진다. 주어진 과정대로 만들어진 1차원 배열을 return 하도록 solution 함수를 완성해라."""

# 2) 제한 사항

"""1 ≤ n ≤ 107
0 ≤ left ≤ right < n2
right - left < 105"""


# 3) 입출력 예시


"""입출력 예 #1



n이 3이고 left가 2이고 right가 5이면, [3,2,2,3]을 반환한다."""

# 4) 코드 설명

"""solution 함수에서 n과 left와 right의 값을 입력 받는다.
결과를 저장하기 위해 result를 빈 리스트로 설정한다.
반복문을 통해 i에 left 값부터 right 값까지 하나씩 할당해준다.
반복문 안에서는 row에 i 값에서 n 값을 나눈 몫 값을 저장하여 i가 속한 행 번호를 저장한다.
col에는 i 값에서 n 값을 나눈 나머지를 저장하여 i가 속한 열 번호를 저장한다.
max 함수를 사용하여 row+1(행)과 col+1(열) 값 중 더 큰 값을 result에 추가한다.
반복문이 끝나면 result 값을 반환한다."""

# 문제 출처 : 프로그래머스
