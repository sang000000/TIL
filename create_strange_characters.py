def solution(s):
    a = []
    b = s.split(" ")
    for i in range(len(b)):
        for j in range(len(b[i])):
            if j % 2 == 0:
                a.append(b[i][j].upper())
            elif j % 2 != 0  :
                a.append(b[i][j].lower())
        a.append(" ")    
    answer = "".join(a[:-1])
    return answer

print(solution("try hello world"))