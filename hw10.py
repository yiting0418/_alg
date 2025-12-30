import numpy as np

def f(*args):
    x = args[0]
    y = args[1]
    z = args[2]
    return 3*x**2 + y**2 + 2*z**2

def r(f, step, *ranges):
    n = len(ranges)
    cnt = 0.0
    
    def recurse(current_args):
        nonlocal cnt
        if len(current_args) == n:
            cnt += f(*current_args) * step**n
            return
        
        r = ranges[len(current_args)]
        for i in np.arange(r[0], r[1], step):
            recurse(current_args + [i])

    recurse([])
    return cnt

def mc(f, N, *ranges):
    n = len(ranges)
    vol = 1.0
    lows = []
    highs = []
    
    for r in ranges:
        vol *= (r[1] - r[0])
        lows.append(r[0])
        highs.append(r[1])
        
    points = np.random.uniform(
        low=np.array(lows)[:, None], 
        high=np.array(highs)[:, None], 
        size=(n, N)
    )
    return vol * np.mean(f(*points))

step = 0.05
N = 1000000
ranges = [[0,1], [0,1], [0,1]]

print("Riemann:", r(f, step, *ranges))
print("Monte Carlo:", mc(f, N, *ranges))