def solution(numbers, target):
    # 1. 미리 종료할 수 있는 예외 처리
    total_sum = sum(numbers)
    if target > total_sum or (total_sum % 2 != target % 2):
        return 0

    # 2. DFS를 통한 탐색
    def dfs(index, current_sum):

        # 종료 조건: 모든 숫자를 처리했을 때
        if index == len(numbers):
            return 1 if current_sum == target else 0

        # 현재 숫자를 더하거나 빼는 경우
        return dfs(index + 1, current_sum + numbers[index]) + dfs(index + 1, current_sum - numbers[index])

    # 초기 호출
    return dfs(0, 0)



#1) 문제 설명

'''n개의 음이 아닌 정수들이 있다. 이 정수들을 순서를 바꾸지 않고 적절히 더하거나 빼서 타겟 넘버를 만들려고 한다. 예를 들어 [1, 1, 1, 1, 1]로 숫자 3을 만들려면 다음 다섯 방법을 쓸 수 있다.
-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3
사용할 수 있는 숫자가 담긴 배열 numbers, 타겟 넘버 target이 매개변수로 주어질 때 숫자를 적절히 더하고 빼서 타겟 넘버를 만드는 방법의 수를 반환하도록 solution 함수를 작성해라.'''


#2) 제한 사항

'''주어지는 숫자의 개수는 2개 이상 20개 이하이다.
각 숫자는 1 이상 50 이하인 자연수이다.
타겟 넘버는 1 이상 1000 이하인 자연수이다.'''


#3) 입출력 예시 



'''입출력 예 #1



n이 [4, 1, 2, 1]이고 target이 4이면,

+4+1-2+1 = 4
+4-1+2-1 = 4
총 2가지 방법이 있으므로, 2를 반환한다.'''


#4) 코드 설명

'''solution 함수에 numbers와 target의 값을 입력 받는다.
예외처리를 하기 위해 total_sum에 numbers의 합을 저장한다.
만약 target의 값이 numbers의 합보다 크면 값을 만들 수가 없고 target은 짝수이고 numbers의 합은 홀수이거나 target은 홀수이고 numbers의 합은 짝수인 경우에도 값을 만들 수 없으므로 0을 반환한다.
그렇지 않은 경우들은 DFS 방식을 사용하여 dfs 함수를 만들어 index 값과 현재까지 계산된 수의 합을 입력 받는다.
함수 안에서는 만약 indxe 값이 numbers 리스트의 길이 값과 같다면, 즉 배열의 끝에 도달했다면 만약 그 때 현재까지 계산된 수의 값이 target과 같다면 1을 반환하고 아니면 0을 반환한다.
같지 않을 때는 재귀 호출 방식을 사용하여 dfs 함수에서 자기 자신을 호출하여 다음 요소 값을 플러스로 한 경우와 마이너스로 한 경우의 수를 구해서 더한다.
solution 함수에서 초기 호출은 인덱스를 0과 현재까지 더한 값을 0으로 해서 시작한다.'''


#문제 출처 : 프로그래머스