def solution(strings, n):
    strings.sort()
    answer = sorted(strings, key=lambda x : x[n])
    return answer



#1) 문제 설명

'문자열로 구성된 리스트 strings와, 정수 n이 주어졌을 때, 각 문자열의 인덱스 n번째 글자를 기준으로 오름차순 정렬하려 한다. 예를 들어 strings가 ["sun", "bed", "car"]이고 n이 1이면 각 단어의 인덱스 1의 문자 "u", "e", "a"로 strings를 정렬한다.'

#2) 제한 사항

'''strings는 길이 1 이상, 50이하인 배열이다.
strings의 원소는 소문자 알파벳으로 이루어져 있다.
strings의 원소는 길이 1 이상, 100이하인 문자열이다.
모든 strings의 원소의 길이는 n보다 크다.
인덱스 1의 문자가 같은 문자열이 여럿 일 경우, 사전순으로 앞선 문자열이 앞쪽에 위치한다.'''

# 3) 입출력 예시

'''입출력 예 #1

"sun", "bed", "car"의 1번째 인덱스 값은 각각 "u", "e", "a" 이다. 이를 기준으로 strings를 정렬하면 ["car", "bed", "sun"] 이다.
입출력 예 #2

"abce"와 "abcd", "cdx"의 2번째 인덱스 값은 "c", "c", "x"이다. 따라서 정렬 후에는 "cdx"가 가장 뒤에 위치한다. "abce"와 "abcd"는 사전순으로 정렬하면 "abcd"가 우선하므로, 답은 ["abcd", "abce", "cdx"] 이다.'''

#4) 코드 설명
'''solution 함수에서 strings와 n의 값을 입력 받는다.
같은 위치에 같은 알파벳이 있는 단어가 있을 수 있으므로 sort() 함수를 이용해 정렬을 먼저 해준다.
sorted() 함수를 이용하여 문자열 리스트 strings를 key 옵션을 사용해 lambda 함수를 이용해 n번째 위치를 기준으로 정렬하여 answer에 저장한다.
그 후 값을 반환한다.'''

#문제 출처 : 프로그래머스