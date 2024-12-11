def solution(arr1, arr2):
    import numpy as np

    # NumPy 배열로 변환
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    # 행렬 곱 계산
    result = np.dot(arr1, arr2)

    # 결과를 NumPy 배열로 반환
    return result.tolist()


# 1) 문제 설명

"""2차원 행렬 arr1과 arr2를 입력받아, arr1에 arr2를 곱한 결과를 반환하는 함수, solution을 완성해라."""

# 2) 제한 사항

"""행렬 arr1, arr2의 행과 열의 길이는 2 이상 100 이하이다.
행렬 arr1, arr2의 원소는 -10 이상 20 이하인 자연수이다.
곱할 수 있는 배열만 주어진다.
"""

# 3) 입출력 예시


"""입출력 예 #1



arr1이 [[1, 4], [3, 2], [4, 1]]이고 arr2가 [[3, 3], [3, 3]]이면, [[15, 15], [15, 15], [15, 15]]을 반환한다."""

# 4) 코드 설명

"""solution 함수에서 arr1, arr2의 값을 입력 받는다.
행렬을 곱하기 위해 numpy를 np라는 이름으로 설정하여 불러온다.
arr1과 arr2를 array 함수를 사용하여 배열 형태로 다시 저장한다.
dot 함수를 사용하여 arr1과 arr2 행렬을 곱하여 result에 저장한다.
그 후 결과를 tolist 함수를 이용해 리스트 형태로 반환한다."""

# 문제 출처 : 프로그래머스
