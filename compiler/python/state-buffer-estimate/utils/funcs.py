def kstr(num):
    return Kstr(num) +"k"

def Kstr(num):
    return str(num // 1024)

def DivCeil(m, n):    ## ceiling of [m/n]
    assert(isinstance(m, int) and isinstance(n, int))
    return (m + n - 1) // n

def DivFloor(m, n):   ## floor of [m/n]
    assert(isinstance(m, int) and isinstance(n, int))
    return  m // n

