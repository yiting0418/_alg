import sys

role  = ['人','狼','羊','菜']

def isDead(state):
    # 狼羊同邊 人在另一邊->狼吃羊
    if state[1]==state[2] and state[0] != state[1]:
        return True
    # 羊菜同邊 人在另一邊->羊吃菜
    if state[2]==state[3] and state[0] != state[2]:
        return True
    return False

def neighbors(state):
    peopleSide = state[0] #紀錄人的位置
    nbList = []

    side2 = 1 if peopleSide==0 else 0 #對岸的數值
    p2state = state.copy()
    p2state[0] = side2 # 人自己移動到另一邊
    nbList.append(p2state)

    for i in range(1, len(state)):
        if state[i] == peopleSide: #如果狼/羊/菜跟人同一邊
            nbState = state.copy()
            nbState[0], nbState[i] = side2, side2 #人帶著它一起移動
            nbList.append(nbState)
    return nbList

def dfs(state, role, visitedMap, goal, path):
    if isDead(state): return #如果有東西被吃
    
    stateStr = ''.join(str(x) for x in state) #轉str
    if visitedMap.get(stateStr): return #檢查此狀態是否重複
    visitedMap[stateStr] = True

    path.append(state) #儲存狀態
    
    if state == goal:
        print("success!")
        #輸出路徑
        for s in path:
            line = ""
            for i in range(4):
                line += f"{role[i]}{s[i]} "
            print(line)
            
        sys.exit(1) 
        return
    #遞迴搜尋
    for nb in neighbors(state):
        dfs(nb, role, visitedMap, goal, path)

    path.pop()


start = [0,0,0,0] #初始值
goal = [1,1,1,1] #目標
visitedMap = {} #紀錄狀態
path = [] #紀錄路徑

# 開始搜尋
dfs(start, role, visitedMap, goal, path)
