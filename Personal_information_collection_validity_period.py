def solution(today, terms, privacies):
    answer = []
    today = list(map(int, today.split(".")))  # today를 정수 리스트로 변환
    term = {}
    
    # terms를 딕셔너리로 변환
    for i in terms:
        t, m = i.split(" ")
        term[t] = int(m)
    
    # privacies를 처리
    for idx, p in enumerate(privacies):
        date, term_type = p.split(" ")
        year, month, day = map(int, date.split("."))
        term_month = term[term_type]
        
        # 만료 날짜 계산
        month += term_month
        
        # 월 넘침 처리
        if month > 12:
            year += month // 12
            month = month % 12
            if month == 0:
                month = 12
                year -= 1
        
        # 만료 날짜와 today 비교
        if (year < today[0]) or (year == today[0] and month < today[1]) or (year == today[0] and month == today[1] and day <= today[2]):
            answer.append(idx + 1)
        
    return sorted(answer)




#1) 문제 설명

'''고객의 약관 동의를 얻어서 수집된 1~n번으로 분류되는 개인정보 n개가 있다. 약관 종류는 여러 가지 있으며 각 약관마다 개인정보 보관 유효기간이 정해져 있다. 당신은 각 개인정보가 어떤 약관으로 수집됐는지 알고 있다. 수집된 개인정보는 유효기간 전까지만 보관 가능하며, 유효기간이 지났다면 반드시 파기해야 한다. 예를 들어, A라는 약관의 유효기간이 12 달이고, 2021년 1월 5일에 수집된 개인정보가 A약관으로 수집되었다면 해당 개인정보는 2022년 1월 4일까지 보관 가능하며 2022년 1월 5일부터 파기해야 할 개인정보이다.
당신은 오늘 날짜로 파기해야 할 개인정보 번호들을 구하려 한다. 모든 달은 28일까지 있다고 가정한다.
다음은 오늘 날짜가 2022.05.19일 때의 예시이다.
   약관 종류                                                      유효기간
A	                                                                6 달
B	                                                                12 달
C	                                                                3 달
번호                                        개인정보 수집 일자                                                                            약관 종류
1	                                          2021.05.02	                                                                                A
2	                                          2021.07.01	                                                                                B
3	                                          2022.02.19	                                                                                C
4	                                          2022.02.20	                                                                                C
첫 번째 개인정보는 A약관에 의해 2021년 11월 1일까지 보관 가능하며, 유효기간이 지났으므로 파기해야 할 개인정보이다.
두 번째 개인정보는 B약관에 의해 2022년 6월 28일까지 보관 가능하며, 유효기간이 지나지 않았으므로 아직 보관 가능하다.
세 번째 개인정보는 C약관에 의해 2022년 5월 18일까지 보관 가능하며, 유효기간이 지났으므로 파기해야 할 개인정보이다.
네 번째 개인정보는 C약관에 의해 2022년 5월 19일까지 보관 가능하며, 유효기간이 지나지 않았으므로 아직 보관 가능하다.
따라서 파기해야 할 개인정보 번호는 [1, 3]이다.
오늘 날짜를 의미하는 문자열 today, 약관의 유효기간을 담은 1차원 문자열 배열 terms와 수집된 개인정보의 정보를 담은 1차원 문자열 배열 privacies가 매개변수로 주어진다. 이때 파기해야 할 개인정보의 번호를 오름차순으로 1차원 정수 배열에 담아 return 하도록 solution 함수를 완성해라.
'''
#2) 제한 사항

'''today는 "YYYY.MM.DD" 형태로 오늘 날짜를 나타낸다.
1 ≤ terms의 길이 ≤ 20
terms의 원소는 "약관 종류 유효기간" 형태의 약관 종류와 유효기간을 공백 하나로 구분한 문자열이다.
약관 종류는 A~Z중 알파벳 대문자 하나이며, terms 배열에서 약관 종류는 중복되지 않는다.
유효기간은 개인정보를 보관할 수 있는 달 수를 나타내는 정수이며, 1 이상 100 이하이다.
1 ≤ privacies의 길이 ≤ 100
privacies[i]는 i+1번 개인정보의 수집 일자와 약관 종류를 나타낸다.
privacies의 원소는 "날짜 약관 종류" 형태의 날짜와 약관 종류를 공백 하나로 구분한 문자열이다.
날짜는 "YYYY.MM.DD" 형태의 개인정보가 수집된 날짜를 나타내며, today 이전의 날짜만 주어진다.
privacies의 약관 종류는 항상 terms에 나타난 약관 종류만 주어진다.
today와 privacies에 등장하는 날짜의 YYYY는 연도, MM은 월, DD는 일을 나타내며 점(.) 하나로 구분되어 있다.
2000 ≤ YYYY ≤ 2022
1 ≤ MM ≤ 12
MM이 한 자릿수인 경우 앞에 0이 붙는다.
1 ≤ DD ≤ 28
DD가 한 자릿수인 경우 앞에 0이 붙는다
파기해야 할 개인정보가 하나 이상 존재하는 입력만 주어진다.'''

#3) 입출력 예시



'''입출력 예 #1

today가 "2020.01.01"이고 terms가 ["Z 3", "D 5"]이고 privacies가 ["2019.01.01 D", "2019.11.15 Z", "2019.08.02 D", "2019.07.01 D", "2018.12.28 Z"]이면
   약관                                                                   종류유효기간
    Z	                                                                     3 달
    D	                                                                     5 달
번호                                      개인정보 수집 일자                                                                             약관 종류
1	                                         2019.01.01	                                                                                    D
2	                                         2019.11.15                                                                                 	Z
3	                                        2019.08.02	                                                                                    D
4	                                        2019.07.01	                                                                                    D
5	                                         2018.12.28	                                                                                    Z
오늘 날짜는 2020년 1월 1일이다.

첫 번째 개인정보는 D약관에 의해 2019년 5월 28일까지 보관 가능하며, 유효기간이 지났으므로 파기해야 할 개인정보이다.
두 번째 개인정보는 Z약관에 의해 2020년 2월 14일까지 보관 가능하며, 유효기간이 지나지 않았으므로 아직 보관 가능하다.
세 번째 개인정보는 D약관에 의해 2020년 1월 1일까지 보관 가능하며, 유효기간이 지나지 않았으므로 아직 보관 가능하다.
네 번째 개인정보는 D약관에 의해 2019년 11월 28일까지 보관 가능하며, 유효기간이 지났으므로 파기해야 할 개인정보이다.
다섯 번째 개인정보는 Z약관에 의해 2019년 3월 27일까지 보관 가능하며, 유효기간이 지났으므로 파기해야 할 개인정보이다.'''

#4) 코드 설명

'''solution 함수에서 today, terms, privacies의 값을 입력 받는다.
값을 저장하기 위해 answer을 빈 리스트로 저장한다.
today를 정수 리스트로 바꾸기 위해 split함수를 이용하여 점을 구분자로하여 분리하고 map함수를 이용하여 각각 정수형으로 바꾼 뒤 list 함수를 이용하여 리스트로 변환하여 today에 다시 저장한다.
terms을 딕셔너리 형태로 바꾸기 위해 빈 딕셔너리를 term에 저장한다.
반복문을 통해 i에 terms 값을 하나씩 집어넣는다.
반복문 안에서는 t와 m에 해당 i의 값을 공백을 구분자로하여 분리하여 각각 저장한다.
term에 해당 t의 값으로 key를 정하고 m을 정수형으로 하여 values 값으로 설정한다.
반복문이 끝나면 다른 반복문이 실행되서 enumerate를 사용하여 idx와 p에 각각 privacies의 인덱스와 값을 넣어준다.
반복문 안에서는 해당 p의 값을 date와 term_type에 각각 공백을 기준으로 나누어 저장한다.
그 후 map 함수를 이용하여 date를 다시 한번 split 함수를 이용하여 점을 기준으로 나누어 각각 정수형으로 year,month.day에 저장한다.
그런 다음 term_montn에다가 term 딕셔너리에서 term_type에 해당하는 value 값을 저장한다.
그리고 만료 날짜를 계산하기 위해 month에 term_month 값을 더해 만료 날짜를 구한다.
만약 month가 12보다 크다면 13이상이므로 만료날짜가 13개월이상라는 소리인데 이는 1년 1개월이상과 같은 뜻이므로 year에 month 나누기 12를 한 값의 몫을 더해준다.
그리고 month는 month 나누기 12를 한 값의 나머지로 바꿔준다.
그때 만약 month가 0이라면 즉 나머지 값이 0이라면 이는 다음 해의 12월이라는 뜻이므로 month는 12로 설정하고 year에서 값을 1 빼준다.
이 과정까지 끝나게 되면 만약 year가 today[0]보다 작다면, 즉 만료 연도가 오늘 연도보다 작거나 혹은 year와 today[0]가 같고 month가 today[1]보다 작다면, 즉 만료 연도와 오늘 연도는 같지만 만료 월이 오늘 월보다 작거나 혹은 year와 today[0]는 같고 month와 today[1]도 같지만 day가 today[2]와 같거나 작다면, 즉 만료 연도와 오늘 연도, 만료 월과 오늘 월도가 같지만 만료 일이 오늘 일보다 작거나 같다는 뜻이므로 위 조건 중 하나라도 만족하게 된다면 만료가 된 것이므로 answer에 해당 인덱스 값 +1을 추가한다.
그 후 answer을 반환한다.'''

#문제 출처 : 프로그래머스


