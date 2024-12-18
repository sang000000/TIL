def solution(number, k):
    stack = []
    count = 0
    
    for i in range(len(number)):
        while stack and count < k and stack[-1] < number[i]:
            stack.pop()
            count += 1
        stack.append(number[i])
    
    # k개의 숫자를 제거했으므로, 남은 숫자에서 더 필요한 만큼 제거
    if count < k:
        stack = stack[:- (k - count)]
    
    answer = "".join(stack)
    return answer



#1) 문제 설명

'''어떤 숫자에서 k개의 수를 제거했을 때 얻을 수 있는 가장 큰 숫자를 구하려 한다.
예를 들어, 숫자 1924에서 수 두 개를 제거하면 [19, 12, 14, 92, 94, 24] 를 만들 수 있다. 이 중 가장 큰 숫자는 94이다.
문자열 형식으로 숫자 number와 제거할 수의 개수 k가 solution 함수의 매개변수로 주어진다. number에서 k 개의 수를 제거했을 때 만들 수 있는 수 중 가장 큰 숫자를 문자열 형태로 반환하도록 solution 함수를 완성하라.
'''

#2) 제한 사항

'''number는 2자리 이상, 1,000,000자리 이하인 숫자입니다.
k는 1 이상 number의 자릿수 미만인 자연수입니다.'''


#3) 입출력 예시 


'''입출력 예 #1



number이 "1231234"이고 k가 3이면, 3개를 제거하여 만들 수 있는 가장 큰 값은 "3234"이다.'''

#3. 코드 설명

'''solution 함수에 number,k 값을 입력 받는다.
가장 큰 숫자를 만들기 위해 stack이라는 빈 리스트를 생성한다.
몇개를 제거했는지 세기 위해 count을 0으로 초기 설정한다.
for문을 통해 0부터 number의 길이 빼기 1까지 i에 값을 하나씩 집어 넣는다.
for문 안에서는 while문을 사용하여 stack이 비어있지 않고 count의 값이 k보다 작고 stack의 마지막 요소 값보다 i번째의 인덱스 위치의 number 요소 값이 더 큰 경우에만 반복한다.
while문 안에서는 while 문의 조건을 만족하면 실행 되므로 stack의 마지막 요소 값이 number의 i번째 인덱스 위치 값보다 작다는 뜻이므로 stack의 마지막 요소 값을 제거한다.
그 후 count의 값을 1 증가시켜 숫자를 몇개 제거했는지 업데이트 한다.
while문이 끝나면 stack에 number의 i번째 인덱스 위치의 값을 추가한다.
for문도 끝이나면 만약 count 값이 k보다 작다면, 아직 제거할 숫자가 남아 있다는 뜻이므로, stack에서 맨 끝에서 k-count 값 만큼 숫자를 추가적으로 제거하여 최종적으로 k값 만큼 숫자를 제거한다.
그 다음 join 함수를 사용하여 stack 리스트를 문자열 형태로 하나로 합친다.
모든 코드가 끝이나면 answer 값을 반환한다.'''

#문제 출처 : 프로그래머스