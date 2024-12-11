def solution(id_list, report, k):
    answer = []
    declaration = {id_list[i]: 0 for i in range(len(id_list))}
    person = {id_list[i]: [] for i in range(len(id_list))}
    report = list(set(report))

    for i in report:
        ide, declare = i.split(" ")
        declaration[declare] += 1
        person[ide].append(declare)

    banned_users = {user for user, count in declaration.items() if count >= k}

    for user in id_list:
        mail_count = 0
        for reported in person[user]:
            if reported in banned_users:
                mail_count += 1
        answer.append(mail_count)
    return answer


# 1) 문제 설명

"""신입사원 무지는 게시판 불량 이용자를 신고하고 처리 결과를 메일로 발송하는 시스템을 개발하려 한다. 무지가 개발하려는 시스템은 다음과 같다.
각 유저는 한 번에 한 명의 유저를 신고할 수 있다.
신고 횟수에 제한은 없습니다. 서로 다른 유저를 계속해서 신고할 수 있다.
한 유저를 여러 번 신고할 수도 있지만, 동일한 유저에 대한 신고 횟수는 1회로 처리된다.
k번 이상 신고된 유저는 게시판 이용이 정지되며, 해당 유저를 신고한 모든 유저에게 정지 사실을 메일로 발송한다.
유저가 신고한 모든 내용을 취합하여 마지막에 한꺼번에 게시판 이용 정지를 시키면서 정지 메일을 발송한다.
다음은 전체 유저 목록이 ["muzi", "frodo", "apeach", "neo"]이고, k = 2(즉, 2번 이상 신고당하면 이용 정지)인 경우의 예시이다.
유저 ID                          유저가 신고한 ID                           설명
"muzi"	                            "frodo"	                      "muzi"가 "frodo"를 신고했다.
"apeach"	                        "frodo"	                      "apeach"가 "frodo"를 신고했다.
"frodo"	                            "neo"	                      "frodo"가 "neo"를 신고했다.
"muzi"	                            "neo"	                      "muzi"가 "neo"를 신고했다.
"apeach"	                        "muzi"	                      "apeach"가 "muzi"를 신고했다.
각 유저별로 신고당한 횟수는 다음과 같다.
유저 ID                                            신고당한 횟수
"muzi"	                                                1
"frodo"	                                                2
"apeach"	                                            0
"neo"	                                                2
위 예시에서는 2번 이상 신고당한 "frodo"와 "neo"의 게시판 이용이 정지된다. 이때, 각 유저별로 신고한 아이디와 정지된 아이디를 정리하면 다음과 같다.
유저 ID                                               유저가 신고한 ID                                         정지된 ID
"muzi"	                                                ["frodo", "neo"]	                                ["frodo", "neo"]
"frodo"	                                                ["neo"]	                                            ["neo"]
"apeach"	                                            ["muzi", "frodo"]	                                ["frodo"]
"neo"	                                                없음	                                            없음
따라서 "muzi"는 처리 결과 메일을 2회, "frodo"와 "apeach"는 각각 처리 결과 메일을 1회 받게 된다. 이용자의 ID가 담긴 문자열 배열 id_list, 각 이용자가 신고한 이용자의 ID 정보가 담긴 문자열 배열 report, 정지 기준이 되는 신고 횟수 k가 매개변수로 주어질 때, 각 유저별로 처리 결과 메일을 받은 횟수를 배열에 담아 반환하도록 solution 함수를 완성해라.
"""

# 2) 제한 사항

"""2 ≤ id_list의 길이 ≤ 1,000
1 ≤ id_list의 원소 길이 ≤ 10
id_list의 원소는 이용자의 id를 나타내는 문자열이며 알파벳 소문자로만 이루어져 있다.
id_list에는 같은 아이디가 중복해서 들어있지 않는다.
1 ≤ report의 길이 ≤ 200,000
3 ≤ report의 원소 길이 ≤ 21
report의 원소는 "이용자id 신고한id"형태의 문자열이다.
예를 들어 "muzi frodo"의 경우 "muzi"가 "frodo"를 신고했다는 의미이다.
id는 알파벳 소문자로만 이루어져 있다.
이용자id와 신고한id는 공백(스페이스)하나로 구분되어 있다.
자기 자신을 신고하는 경우는 없다.
1 ≤ k ≤ 200, k는 자연수이다.
return 하는 배열은 id_list에 담긴 id 순서대로 각 유저가 받은 결과 메일 수를 담으면 된다."""


# 3) 입출력 예시


"""입출력 예 #1

id_list가 ["con", "ryan"]이고 report가 ["ryan con", "ryan con", "ryan con", "ryan con"]이고 k가 3이면, "ryan"이 "con"을 4번 신고했으나, 주어진 조건에 따라 한 유저가 같은 유저를 여러 번 신고한 경우는 신고 횟수 1회로 처리한다. 따라서 "con"은 1회 신고당했다. 3번 이상 신고당한 이용자는 없으며, "con"과 "ryan"은 결과 메일을 받지 않는다. 따라서 [0, 0]을 반환한다."""

# 4) 코드 설명

"""solution 함수에서 id_list, report, k의 값을 입력 받는다.
값을 저장하기 위해 answer이라는 빈 리스트를 만든다.
신고된 횟수를 세기 위해서 id_list의 각 요소를 key로 하고 value는 초기 값을 0으로 하는 declaration이라는 딕셔너리를 생성한다.
누가 누구를 신고했지는 알기 위해서 id_list의 각 요소를 key로 하고 value는 초기 값은 []로 하는 person이라는 딕셔너리를 생성한다.
한 사람이 같은 사람이 계속 신고해도 1번 밖에 인정이 안되므로 report를 set을 사용하여 중복을 제거한다.
반복문을 통해 report의 값을 하나씩 i에 집어넣어 ide와 declare에 해당 i 값을 공백을 기준으로 분리하여 각각 저장한다.
그 후 declaration 딕셔너리에 해당 key 값에 값을 1 증가시키고 person 딕셔너리에 해당 key 값에 value 리스트에 신고한 사람을 적는다.
반복문이 끝나면 신고 당한 횟수가 k값보다 크거나 같은 사람의 key값만 가져와서 집합으로 만든다.
그 다음에 다른 반복문을 통해 id_list의 각 요소는 user에 하나씩 집어넣어 반복한다.
반복문 안에서는 처음에 mail_count를 0으로 초기화해준다.
그 다음에 반복문을 또 실행하여 person 딕셔너리 user에 해당하는 value 값에 reported에 하나씩 집어넣고 그 안에서 만약 reported가 banned_users에 있다면, 즉 reported 유저가 정지된 유저 목록(banned_users)에 포함되어 있다면, mail_count 값을 1 증가 시킨다.
그 후 반복문 안에 있는 반복문이 끝나면 answer에 그 값을 추가한다.
그 후 모든 반복문이 종료 되면 answer 값을 반환한다."""

# 문제 출처 : 프로그래머스
