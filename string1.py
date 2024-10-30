def solution(s):
    a = []
    b = ''
    c1 = 1
    c2 = 0

    while s:  
        x = s[0]  
        b += x

        for i in range(1, len(s)):
            print("i", i)
            if x == s[i]:
                b += s[i]
                c1 += 1
                print("c1", c1)
            else:
                b += s[i]
                c2 += 1
                print("c2", c2)
            if c1 == c2:                
                break

        a.append(b)
        s = s.replace(b, "", 1).strip()
        b = ''
        c1 = 1
        c2 = 0
        
    answer = len(a)
    return answer+



#1) 문제 설명

'''문자열 s가 입력되었을 때 다음 규칙을 따라서 이 문자열을 여러 문자열로 분해하려고 한다.
먼저 첫 글자를 읽는다. 이 글자를 x라고 한다.
이제 이 문자열을 왼쪽에서 오른쪽으로 읽어나가면서, x와 x가 아닌 다른 글자들이 나온 횟수를 각각 센다. 처음으로 두 횟수가 같아지는 순간 멈추고, 지금까지 읽은 문자열을 분리한다.
s에서 분리한 문자열을 빼고 남은 부분에 대해서 이 과정을 반복한다. 남은 부분이 없다면 종료한다.
만약 두 횟수가 다른 상태에서 더 이상 읽을 글자가 없다면, 역시 지금까지 읽은 문자열을 분리하고, 종료한다.
문자열 s가 매개변수로 주어질 때, 위 과정과 같이 문자열들로 분해하고, 분해한 문자열의 개수를 return 하는 함수 solution을 완성하라.'''

#2) 제한 사항

'''1 ≤ s의 길이 ≤ 10,000
s는 영어 소문자로만 이루어져 있다.'''

#3) 입출력 예시



'''입출력 예 #1

s="banana"인 경우 ba - na - na와 같이 분해된다.


입출력 예 #2

s="abracadabra"인 경우 ab - ra - ca - da - br - a와 같이 분해된다.
입출력 예 #3

s="aaabbaccccabba"인 경우 aaabbacc - ccab - ba와 같이 분해된다.'''

#4) 코드 설명

'''solution 함수에서 s의 값을 입력 받는다.
분해한 문자열의 개수를 세기 위해 빈 리스트 a를 만든다.
문자열을 분해하기 위해 빈 문자열 b를 만든다.
첫 문자부터 세기 때문에 무조건 하나는 있기 때문에 c1은 초기 값은 1로 설정한다.
첫 문자 다음부터 세기 위해 c2는 0으로 설정한다.
while문을 사용해 s가 비어있지 않을 때까지 반복한다.
while문 안에서는 첫 글자부터 세기 위해 x에 문자열 s의 첫 글자를 저장한다.
문자열의 첫 글자는 나눌 때 무조건 들어 있어야 하므로 b에 x값을 추가한다.
그 후 for문을 실행해서 1부터 문자열 s의 길이 빼기 1값까지 i에 집어 넣어는다.
for문 안에서는 만약 x값이 문자열 s의 해당 i 값 위치의 문자와 같으면 b에 그 값을 추가한다.
그 후 c1의 값을 1 추가한다.
만약 x와 s[i]의 값이 다르다면 b에 문자열 s의 해당 i 값 위치의 문자를 추가한다.
그러다가 만약 c1과 c2의 값이 같다면 for문을 종료한다.
그 후 b의 값을 리스트 a에 추가한다.
또 맨 처음 나오는 b의 값을 빈 문자열로 교체하고 strip() 함수로 빈 문자열을 삭제한다.
그 후 문자열의 나머지 부분도 똑같이 진행하기 위해 b는 다시 빈 문자열로 만들고 c1은 1로 하고 c2는 0으로 다시 설정한다.
그러다가 문자열 s를 모두 분리하여 빈 상태가 되면 while문이 끝나고 answer에다가 리스트 a의 길이를 측정한 후 저장한다.
마지막으로 그 값을 반환한다. '''

#문제 출처 : 프로그래머스