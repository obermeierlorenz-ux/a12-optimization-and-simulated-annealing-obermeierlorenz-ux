import numpy as np  
import matplotlib.pyplot as plt


FEATURES = np.array([
[-5.0, -1.85, -1.35, 0.48, 0.62],
[-4.3, 0.15, -1.80, 0.65, 0.38],
[-6.0, 1.65, -0.70, 0.38, 0.55],
[-4.8, -1.30, 0.75, 0.58, 0.42],
[-5.5, 0.15, 1.75, 0.40, 0.52],
[-3.9, 1.95, 1.45, 0.55, 0.38],
[ 3.5, -0.65, -0.65, 0.42, 0.48],
[ 4.0, 0.70, 0.45, 0.45, 0.38],
[ 3.0, -1.95, 1.85, 0.34, 0.40],
[ 2.8, 1.30, -2.05, 0.32, 0.35],
])

def function2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = 0.06 * (x**2 + y**2)
    for amplitude , cx, cy, sx, sy in FEATURES:
        exponent = ((x - cx) / sx)**2 + ((y - cy) / sy)**2
        z = z  + amplitude * np.exp(-0.5 * exponent)
    return z


def function1(x, y):
    return(x**2 - 1)**2 + 0.2*x + 4*(y-0.5*x**2)**2


def f_test(x, y):
    return  x**2 + y**2
