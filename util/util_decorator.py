from concurrent.futures import ProcessPoolExecutor 
from functools import wraps
def mp_dcr(tgt):
    @wraps(tgt)
    def wrapper(*args, **kwargs):
        with ProcessPoolExecutor() as pool:
            future = pool.map(tgt( zip(*args), zip(**kwargs)))
            return future
    return wrapper


@mp_dcr
def f(x,y):
    s = x + y 
    print(s)
    return s 