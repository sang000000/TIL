def solution(players, callings):
    # 선수의 이름과 인덱스를 매핑하는 딕셔너리 생성
    index_map = {player: idx for idx, player in enumerate(players)}

    for calling in callings:
        idx = index_map[calling]  # 호출된 선수의 현재 인덱스를 찾음

        # 스왑할 선수
        if idx > 0:  # 첫 번째 선수는 스왑할 수 없음
            # 스왑
            players[idx], players[idx - 1] = players[idx - 1], players[idx]

            # 인덱스 업데이트
            index_map[calling] = idx - 1
            index_map[players[idx]] = idx  # 스왑된 선수의 인덱스 업데이트

    return players


# 1) 문제 설명

"""얀에서는 매년 달리기 경주가 열린다. 해설진들은 선수들이 자기 바로 앞의 선수를 추월할 때 추월한 선수의 이름을 부른다. 예를 들어 1등부터 3등까지 "mumu", "soe", "poe" 선수들이 순서대로 달리고 있을 때, 해설진이 "soe"선수를 불렀다면 2등인 "soe" 선수가 1등인 "mumu" 선수를 추월했다는 것이다. 즉 "soe" 선수가 1등, "mumu" 선수가 2등으로 바뀐다.
선수들의 이름이 1등부터 현재 등수 순서대로 담긴 문자열 배열 players와 해설진이 부른 이름을 담은 문자열 배열 callings가 매개변수로 주어질 때, 경주가 끝났을 때 선수들의 이름을 1등부터 등수 순서대로 배열에 담아 반환하는 solution 함수를 완성해라.
"""
# 2) 제한 사항

"""5 ≤ players의 길이 ≤ 50,000
players[i]는 i번째 선수의 이름을 의미합니다.
players의 원소들은 알파벳 소문자로만 이루어져 있습니다.
players에는 중복된 값이 들어가 있지 않습니다.
3 ≤ players[i]의 길이 ≤ 10
.2 ≤ callings의 길이 ≤ 1,000,000
callings는 players의 원소들로만 이루어져 있습니다.
경주 진행중 1등인 선수의 이름은 불리지 않습니다."""

# 3) 입출력 예시


"""players가 ["mumu", "soe", "poe", "kai", "mine"]이고 callings가 ["kai", "kai", "mine", "mine"]이면, 4등인 "kai" 선수가 2번 추월하여 2등이 되고 앞서 3등, 2등인 "poe", "soe" 선수는 4등, 3등이 된다. 5등인 "mine" 선수가 2번 추월하여 4등, 3등인 "poe", "soe" 선수가 5등, 4등이 되고 경주가 끝난다. 1등부터 배열에 담으면 ["mumu", "kai", "mine", "soe", "poe"]이 된다."""

# 4) 코드 설명

"""solution 함수에서 players, callings의 값을 입력 받는다.
소요 시간을 줄이기 위해 index_map이라는 딕셔너리를 만들어 그 안에 enumerate 함수를 사용하여 idx와 player에 각각 인덱스와 플레이어 이름을 집어넣어 player가 key이고 idx가 values값이 되게 반복해서 저장한다.
그 후 반복문을 통해 callings의 값을 calling에 하나씩 집어넣고 idx에 index_map 딕셔너리에서 해당 calling의 값에 해당하는 values값을 가져와 저장한다.
만약 idx가 0보다 크면 1등은 아니라는 소리이므로 제칠 사람이 있단는 뜻이라 players 리스트에서 해당 idx 위치에 선수와 players 리스트에서 해당 idx 빼기 1을 한 위치에 선수를 서로 바꾸어 저장한다.
그 후 index_map 딕셔너리의 calling key의 value를 idx-1한 값으로 바꿔주고
index_map 딕셔너리에서 역전당한 선수는 idx로 바꾸어 준다.
반복문이 끝나면 players 리스트를 반환한다."""

# 문제 출처 : 프로그래머스
