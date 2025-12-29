def min_edit_distance_verbose(word1, word2):
    """
    計算最小編輯距離，並印出詳細的推論過程 (Verbose Version)
    """
    m = len(word1)
    n = len(word2)
    
    # 建立表格
    # dp[i][j] 代表 word1 的前 i 個字變到 word2 的前 j 個字
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # --- 步驟 1: 初始化邊框 (填第一列與第一欄) ---
    print(f"=== 1. 初始化表格邊框 ===")
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    print("邊框初始化完成 (第一列代表全插入，第一欄代表全刪除)。\n")

    # --- 步驟 2: 開始填中間的格子 ---
    print(f"=== 2. 開始填表推論 ===")
    
    # 為了版面好看，我們定義一下動作名稱
    # 上面(Top) = 刪除 word1 的字
    # 左邊(Left) = 插入 word2 的字
    # 左上(Diag) = 替換 (或是相同)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            char1 = word1[i-1]
            char2 = word2[j-1]
            
            print(f"--------------------------------------------------")
            print(f"正在計算 dp[{i}][{j}] ( '{char1}' -> '{char2}' )")

            # 情況 A: 字元相同
            if char1 == char2:
                dp[i][j] = dp[i-1][j-1]
                print(f"  [判定] 字元相同！")
                print(f"  [動作] 直接繼承左上角 (Pass/Match)")
                print(f"  [數值] {dp[i-1][j-1]} (不加分)")
            
            # 情況 B: 字元不同，需要操作
            else:
                delete_cost = dp[i-1][j]    # 上面的值
                insert_cost = dp[i][j-1]    # 左邊的值
                sub_cost    = dp[i-1][j-1]  # 左上角的值

                # 找出最小的代價
                min_val = min(delete_cost, insert_cost, sub_cost)
                dp[i][j] = 1 + min_val

                print(f"  [判定] 字元不同！檢查三個鄰居：")
                print(f"    - 上 (刪除 '{char1}'): {delete_cost}")
                print(f"    - 左 (插入 '{char2}'): {insert_cost}")
                print(f"    - 左上 (替換): {sub_cost}")
                
                # 判斷具體是哪一個動作導致了最小值 (僅供顯示用)
                actions = []
                if min_val == sub_cost: actions.append("替換 (Substitute)")
                if min_val == delete_cost: actions.append("刪除 (Delete)")
                if min_val == insert_cost: actions.append("插入 (Insert)")
                
                chosen_action = actions[0] # 簡單起見，取第一個符合的
                
                print(f"  [選擇] 最小鄰居是 {min_val}，動作是: 【{chosen_action}】")
                print(f"  [計算] {min_val} + 1 = {dp[i][j]}")

    # --- 步驟 3: 印出最終表格 ---
    print(f"\n=== 3. 最終表格 (DP Table) ===")
    # 印出表頭
    print("      " + " ".join([f"{c:2}" for c in " #"+word2]))
    for i in range(m + 1):
        row_char = word1[i-1] if i > 0 else '#'
        row_vals = " ".join([f"{val:2}" for val in dp[i]])
        print(f"{row_char:2} [ {row_vals} ]")

    return dp[m][n]

# --- 測試區 ---
if __name__ == "__main__":
    # 建議先用短一點的字測試，方便閱讀
    s1 = "horse"
    s2 = "ros"
    
    print(f"將 '{s1}' 轉換成 '{s2}' 的過程：\n")
    dist = min_edit_distance_verbose(s1, s2)
    print(f"\n最終最小編輯距離: {dist}")
