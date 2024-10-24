import datetime
def solution(a, b):
    weekdays = {0 : "MON", 1 : "TUE", 2: "WED", 3 : "THU", 4 : "FRI", 5 : "SAT", 6 : "SUN"}
    day = datetime.date(2016, a, b)
    print(day)
    print(day.weekday())
    answer = weekdays.get(day.weekday())
    return answer



#1) 문제 설명

'2016년 1월 1일은 금요일이다. 2016년 a월 b일은 무슨 요일일까? 두 수 a ,b를 입력받아 2016년 a월 b일이 무슨 요일인지 리턴하는 함수, solution을 완성하라. 요일의 이름은 일요일부터 토요일까지 각각 SUN,MON,TUE,WED,THU,FRI,SAT이다. 예를 들어 a=5, b=24라면 5월 24일은 화요일이므로 문자열 "TUE"를 반환하라.'

#2) 제한 사항

'''2016년은 윤년이다.
2016년 a월 b일은 실제로 있는 날이다. (13월 26일이나 2월 45일같은 날짜는 주어지지 않는다)'''

#3) 입출력 예시

'''입출력 예 #1

a가 5이고 b가 24이면, "THU"를 반환한다.'''

#4) 코드 설명

'''요일을 쉽게 확인하기 위해 datetime 모듈을 가져온다.
solution 함수에서 a와 b의 값을 입력 받는다.
weekdays라는 요일을 적은 딕셔너리를 만든다.
date라는 함수를 이용해 a와 b값을 입력 받아 날짜 데이터를 만들어 day에 저장한다.
weekday()라는 함수를 이용하여 day에 만든 날짜 데이터를 확인하여 요일을 숫자 값으로 가져오고 그 숫자 값을 get 함수를 이용해 weekday라는 딕셔너리 key값으로 사용하여 날짜를 가져와 answer에 저장한다.
그 후 값을 반환한다.'''

#문제 출처 : 프로그래머스