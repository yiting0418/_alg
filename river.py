import sys

role  = ['人','狼','羊','菜']

def isDead(state):
    # 狼吃羊 (人在另一邊)
    if state[1]==state[2] and state[0] != state[1]:
        return True
    # 羊吃菜 (人在另一邊)
    if state[2]==state[3] and state[0] != state[2]:
        return True
    return False

def neighbors(state):
    peopleSide = state[0]
    nbList = []

    side2 = 1 if peopleSide==0 else 0
    p2state = state.copy()
    p2state[0] = side2 # 人自己移動到另一邊
    nbList.append(p2state)

    for i in range(1, len(state)):
        if state[i] == peopleSide: # 如果跟人同一邊
            nbState = state.copy()
            nbState[0], nbState[i] = side2, side2 # 人帶東西移動
            nbList.append(nbState)
    return nbList

def dfs(state, role, visitedMap, goal, path):
    if isDead(state): return
    
    stateStr = ''.join(str(x) for x in state)
    if visitedMap.get(stateStr): return
    visitedMap[stateStr] = True

    path.append(state)
    
    if state == goal:
        print("success!")
        for s in path:
            line = ""
            for i in range(4):
                line += f"{role[i]}{s[i]} "
            print(line)
            
        sys.exit(1) 
        return

    for nb in neighbors(state):
        dfs(nb, role, visitedMap, goal, path)

    path.pop()


start = [0,0,0,0]
goal = [1,1,1,1]
visitedMap = {}
path = []

# 開始搜尋
dfs(start, role, visitedMap, goal, path)