import numpy as np
g = 9.8 #𝑔 = 9.8 𝑚. 𝑠 ^−2
l = 1.0 #𝑙 = 1.0 𝑚

def subdiv_reg(a, b, n):

    h = (b - a) / n
    return [ a + k * h for k in range(n + 1) ]

def f(x):
    return x**2

def int_rectangle(f : callable([[float], float]), sub_div_list):
    integral = 0
    N = len(sub_div_list) - 1
    for i in range(N):
        integral += f(sub_div_list[i])
    return integral * (sub_div_list[1] - sub_div_list[0])
