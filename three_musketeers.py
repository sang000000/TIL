def solution(number):
    answer = 0
    for i in range(len(number) - 2):
        for j in range(i + 1, len(number) - 1):
            for k in range(j + 1, len(number)):
                if number[i] + number[k] + number[j] == 0:
                    answer += 1
    return answer


# 1) 문제 설명

"""한국중학교에 다니는 학생들은 각자 정수 번호를 갖고 있다. 이 학교 학생 3명의 정수 번호를 더했을 때 0이 되면 3명의 학생은 삼총사라고 한다. 예를 들어, 5명의 학생이 있고, 각각의 정수 번호가 순서대로 -2, 3, 0, 2, -5일 때, 첫 번째, 세 번째, 네 번째 학생의 정수 번호를 더하면 0이므로 세 학생은 삼총사이다. 또한, 두 번째, 네 번째, 다섯 번째 학생의 정수 번호를 더해도 0이므로 세 학생도 삼총사이다. 따라서 이 경우 한국중학교에서는 두 가지 방법으로 삼총사를 만들 수 있다.
한국중학교 학생들의 번호를 나타내는 정수 배열 number가 매개변수로 주어질 때, 학생들 중 삼총사를 만들 수 있는 방법의 수를 return 하도록 solution 함수를 완성하라."""

# 2) 제한 사항

"""3 ≤ number의 길이 ≤ 13
-1,000 ≤ number의 각 원소 ≤ 1,000
서로 다른 학생의 정수 번호가 같을 수 있다."""

# 3) 입출력 예시

"""입출력 예 #1

number이 [-3,-2,-1,0,1,2,3]이면 학생들의 정수 번호 쌍 (-3, 0, 3), (-2, 0, 2), (-1, 0, 1), (-2, -1, 3), (-3, 1, 2) 이 삼총사가 될 수 있으므로, 5를 return 한다.
입출력 예 #2

number이 [-1,1,-1,1]이면 삼총사가 될 수 있는 방법이 없다."""

# 4. 코드 설명

"""solution 함수에서 number의 값을 입력 받는다.

방법의 수를 세기 위해 answer은 초기 값을 0으로 준다.

세 번에 반복문을 통해 리스트에 앞에 있는 요소부터 차례대로 경우의 수를 찾기 위해 먼저 i가 0부터 number의 길이 뺴기 2만큼 차례대로 증가시킨다.

두 번째 반복문에서는 앞에 i를 통해 숫자가 정해졌으므로 그것을 제외한 i+1부터 number의 길이 빼기 1만큼 차례대로 증가시킨다.

세 번쨰 반복문에서는 앞에서 i와 j를 통해 숫자가 정해졌으므로 그것들을 제외한 k+1부터 number의 길이 끝까지 차례대로 증가시킨다.

위에 세 반복문들을 통해 만약 i, j, k 값을 바꿔가면 더한 값이 0이라면 문제의 조건을 만족하게 되는 것이므로 answer의 값을 1 증가시킨다.

모든 반복문이 끝나면 그 떄의 answer 값을 반환하여 몇가지 방법이 있는지 알아낼 수 있다."""

# 문제 출처 : 프로그래머스
