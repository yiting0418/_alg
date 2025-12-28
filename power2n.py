pow = [None]*10000
pow[0] = 1
# 方法 1
def power2n_a(n):
    return 2**n

# 方法 2a：用遞迴
def power2n_b(n):
    if n == 0: return 1
    return power2n_b(n-1)+power2n_b(n-1)

# 方法2b：用遞迴
def power2n_c(n):
    # pass
    if n == 0: return 1
    return 2*power2n_c(n-1)

# 方法 3：用遞迴+查表
def power2n_d(n):
    if n < 0: raise
    if not pow[n] is None: return pow[n]
    pow[n] = power2n_d(n-1)+power2n_d(n-1) 
    return pow[n]

print('power2n(10)=', power2n_c(40))
print('\npower2n_c(40)=',power2n_d(40))