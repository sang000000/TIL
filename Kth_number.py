def solution(array, commands):
    answer = []
    for m in range(len(commands)):
        i, j, k = map(int, commands[m])
        a = array[i-1:j]
        a.sort()
        answer.append(a[k-1])        
    return answer



#1) 문제 설명

'''배열 array의 i번째 숫자부터 j번째 숫자까지 자르고 정렬했을 때, k번째에 있는 수를 구하려 한다.
예를 들어 array가 [1, 5, 2, 6, 3, 7, 4], i = 2, j = 5, k = 3이라면
array의 2번째부터 5번째까지 자르면 [5, 2, 6, 3]이다.
1에서 나온 배열을 정렬하면 [2, 3, 5, 6]이다.
2에서 나온 배열의 3번째 숫자는 5이다.
배열 array, [i, j, k]를 원소로 가진 2차원 배열 commands가 매개변수로 주어질 때, commands의 모든 원소에 대해 앞서 설명한 연산을 적용했을 때 나온 결과를 배열에 담아 return 하도록 solution 함수를 작성해라.'''

#2) 제한 사항

'''array의 길이는 1 이상 100 이하이다.
array의 각 원소는 1 이상 100 이하이다.
commands의 길이는 1 이상 50 이하이다.
commands의 각 원소는 길이가 3이다.'''

#3) 입출력 예시

'''입출력 예 #1

array가 [1, 5, 2, 6, 3, 7, 4]이고 commands가 [[2, 5, 3], [4, 4, 1], [1, 7, 3]이면,
[1, 5, 2, 6, 3, 7, 4]를 2번째부터 5번째까지 자른 후 정렬한다. [2, 3, 5, 6]의 세 번째 숫자는 5이다.
[1, 5, 2, 6, 3, 7, 4]를 4번째부터 4번째까지 자른 후 정렬한다. [6]의 첫 번째 숫자는 6이다.
[1, 5, 2, 6, 3, 7, 4]를 1번째부터 7번째까지 자른다. [1, 2, 3, 4, 5, 6, 7]의 세 번째 숫자는 3이다.'''

#4) 코드 설명
'''solution 함수에서 array와 commands의 값을 입력 받는다.
값을 저장하기 위한 answer이라는 빈 리스트를 만든다.
i, j, k의 값을 map 함수를 이용해 정수형태로 commands 리스트 안의 리스트 값들을 하나씩 집어 넣는다.
인덱스를 활용하여 인덱스는 0부터 시작하므로 array[i-1:j]으로 설정하여 i번째 위치부터 j번째까지 리스트를 잘라내고 그 값을 a에 저장한다.
그 후 오름차순으로 정렬을 한다.
잘라내어 만든 리스트에서 인덱스를 이용하여 a[k-1]으로 설정하여 k번째의 값을 answer 리스트에 추가한다.
위 3~6의 과정을 반복문을 통해 commands의 길이 만큼 반복한다.
반복문이 끝나면 answer 리스트를 반환한다.'''

#문제 출처 : 프로그래머스
