def solution(wallpaper):
    answer = []
    left = len(wallpaper[0])  # 최대 길이로 초기화
    right = 0
    top = len(wallpaper)  # 최대 길이로 초기화
    bottom = 0

    for i in range(len(wallpaper)):
        if "#" in wallpaper[i]:
            top = min(top, i)  # 가장 위쪽 인덱스
            bottom = i  # 가장 아래쪽 인덱스

            # 왼쪽과 오른쪽 인덱스 계산
            left = min(left, wallpaper[i].index("#"))
            right = max(right, wallpaper[i].rindex("#") + 1)  # +1을 해서 포함시킴


    answer.append(top)
    answer.append(left)
    answer.append(bottom + 1)  # bottom은 포함되지 않으므로 +1
    answer.append(right)    
    
    return answer




#1) 문제 설명

'''코딩테스트를 준비하는 머쓱이는 프로그래머스에서 문제를 풀고 나중에 다시 코드를 보면서 공부하려고 작성한 코드를 컴퓨터 바탕화면에 아무 위치에나 저장해 둔다. 저장한 코드가 많아지면서 머쓱이는 본인의 컴퓨터 바탕화면이 너무 지저분하다고 생각했다. 프로그래머스에서 작성했던 코드는 그 문제에 가서 다시 볼 수 있기 때문에 저장해 둔 파일들을 전부 삭제하기로 했다.
컴퓨터 바탕화면은 각 칸이 정사각형인 격자판이다. 이때 컴퓨터 바탕화면의 상태를 나타낸 문자열 배열 wallpaper가 주어진다. 파일들은 바탕화면의 격자칸에 위치하고 바탕화면의 격자점들은 바탕화면의 가장 왼쪽 위를 (0, 0)으로 시작해 (세로 좌표, 가로 좌표)로 표현한다. 빈칸은 ".", 파일이 있는 칸은 "#"의 값을 가진다. 드래그를 하면 파일들을 선택할 수 있고, 선택된 파일들을 삭제할 수 있다. 머쓱이는 최소한의 이동거리를 갖는 한 번의 드래그로 모든 파일을 선택해서 한 번에 지우려고 하며 드래그로 파일들을 선택하는 방법은 다음과 같다. 
점 S(lux, luy)에서 점 E(rdx, rdy)로 드래그를 할 때, "드래그 한 거리"는 |rdx - lux| + |rdy - luy|로 정의한다.
점 S에서 점 E로 드래그를 하면 바탕화면에서 두 격자점을 각각 왼쪽 위, 오른쪽 아래로 하는 직사각형 내부에 있는 모든 파일이 선택된다.
예를 들어 wallpaper = [".#...", "..#..", "...#."]이면, S(0, 1)에서 E(3, 4)로 드래그하면 세 개의 파일이 모두 선택되므로 드래그 한 거리 (3 - 0) + (4 - 1) = 6을 최솟값으로 모든 파일을 선택 가능하다.

(0, 0)에서 (3, 5)로 드래그해도 모든 파일을 선택할 수 있지만 이때 드래그 한 거리는 (3 - 0) + (5 - 0) = 8이고 이전의 방법보다 거리가 늘어난다.드래그는 바탕화면의 격자점 S(lux, luy)를 마우스 왼쪽 버튼으로 클릭한 상태로 격자점 E(rdx, rdy)로 이동한 뒤 마우스 왼쪽 버튼을 떼는 행동이다. 이때, "점 S에서 점 E로 드래그한다"고 표현하고 점 S와 점 E를 각각 드래그의 시작점, 끝점이라고 표현한다.
머쓱이의 컴퓨터 바탕화면의 상태를 나타내는 문자열 배열 wallpaper가 매개변수로 주어질 때 바탕화면의 파일들을 한 번에 삭제하기 위해 최소한의 이동거리를 갖는 드래그의 시작점과 끝점을 담은 정수 배열을 return하는 solution 함수를 작성해 라. 드래그의 시작점이 (lux, luy), 끝점이 (rdx, rdy)라면 정수 배열 [lux, luy, rdx, rdy]를 반환하면 된다.
'''
#2) 제한 사항

'''1 ≤ wallpaper의 길이 ≤ 50
1 ≤ wallpaper[i]의 길이 ≤ 50
wallpaper의 모든 원소의 길이는 동일하다.
wallpaper[i][j]는 바탕화면에서 i + 1행 j + 1열에 해당하는 칸의 상태를 나타낸다.
wallpaper[i][j]는 "#" 또는 "."의 값만 가진다.
바탕화면에는 적어도 하나의 파일이 있다.
드래그 시작점 (lux, luy)와 끝점 (rdx, rdy)는 lux < rdx, luy < rdy를 만족해야 한다.'''

#3) 입출력 예시



'''입출력 예 #1

wallpaper가 ["..........", ".....#....", "......##..", "...##.....", "....#....."]이면 바탕화면은 다음과 같다.(1, 3)에서 (5, 8)로 드래그하면 모든 파일을 선택할 수 있고 이보다 적은 이동거리로 모든 파일을 선택하는 방법은 없다. 따라서 가장 적은 이동의 드래그로 모든 파일을 선택하는 방법인 [1, 3, 5, 8]을 반환한다.
'''

#4) 코드 설명

'''solution 함수에서 wallpaper의 값을 입력 받는다.
값을 반환하기 위해 answer이라는 빈 리스트를 만든다.
시작점과 끝점을 구하기위해 left는 초기 값을 wallpaper의 요소 값의 최대길이로 설정하고 top은 wallpaper의 전체 길이로 설정하고 right와 bottom은 0으로 설정한다.
반복문을 통해 i에 0부터 wallpaper의 길이 빼기 1까지 i에 하나씩 집어 넣는다.
반복문 안에서는 top은 min 함수를 사용하여 기존에 저장된 top값과 i값을 비교하여 더 작은 값으로 바꿔준다.
bottom은 파일이 있는 가장 아래 위치이므로 i의 값으로 바꿔준다.
left는 min함수를 사용하여 기존에 저장된 left값과 wallpaper에 해당 요소값의 "#"위치와 비교하여 더 작은 값으로 바꿔준다.
right는 max 함수를 사용하여 기존에 저장된 right값과 wallpaper에 해당 요소값의 "#"위치를 뒤어서부터 찾아서 1 더한 값과 비교하여 더 큰 값으로 바꾸어 준다.
반복문이 종료 후에 top,left,bottom+1,right 순으로 answer에 추가해준다.
그 후 answer을 반환한다.'''

#문제 출처 : 프로그래머스
