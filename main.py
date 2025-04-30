import math
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


g = 9.8  # 𝑔 = 9.8 𝑚. 𝑠 ^−2
l = 1.0  # 𝑙 = 1.0 𝑚


def subdiv_reg(a, b, n):
    h = (b - a) / n
    return [a + k * h for k in range(n + 1)]


def f(x):
    return x ** 2


def int_rectangle(f: callable([[float], float]), sub_div_list):
    integral = 0
    N = len(sub_div_list) - 1
    for i in range(N):
        integral += f(sub_div_list[i])
    return integral * (sub_div_list[1] - sub_div_list[0])


def int_trapez(f: callable([[float], float]), sub_div_list):
    integral = 0
    N = len(sub_div_list) - 1
    for i in range(N):
        integral += (f(sub_div_list[i]) + f(sub_div_list[i + 1])) / 2
    return integral * (sub_div_list[1] - sub_div_list[0])


def f2(x):
    return np.sqrt(1 - x ** 2)


def periode(theta_max):
    """
    T(θₘₐₓ) = 2 √2ℓ/g ∫₀^{θₘₐₓ} dθ / √(cos θ − cos θₘₐₓ)
    """
    integral = lambda theta: 1 / np.sqrt(np.cos(theta) - np.cos(theta_max))
    res = 2 * math.sqrt(2 * l / g) * int_rectangle(integral, subdiv_reg(0, theta_max, 100))
    return res
def trace_periode():
    thetas = np.linspace(0.1, 3.14, 500)
    Ts = [periode(th) for th in thetas]

    plt.figure()
    plt.plot(thetas, Ts)
    plt.xlabel(r'$\theta_{\max}$ (rad)')
    plt.ylabel('Période T (s)')
    plt.title('Période du pendule en fonction de θ_max')
    plt.grid(True)
    plt.show()


def methode_Euler(
    x0: float,
    y0: float,
    xN: float,
    N: int,
    g: Callable[[float, float], float]
) -> List[Tuple[float, float]]:
    h = (xN - x0) / N
    xs = [x0 + i * h for i in range(N+1)]
    ys = [0.0] * (N+1)
    ys[0] = y0

    for i in range(N):
        ys[i+1] = ys[i] + h * g(xs[i], ys[i])

    return list(zip(xs, ys))



inputVal = int(input("Type 1 to show 𝐼 = ∫₀¹ √(1 − x²) dx = π/4 \n Type 2 to show T(θₘₐₓ) = 2 √2ℓ/g ∫₀^{θₘₐₓ} dθ / √(cos θ − cos θₘₐₓ) \n Type 3 to show the graph of T(θₘₐₓ) \n"))
if (inputVal == 1):
    # Integral of f(x) from 0 to 1 using rectangles and trapez
    rec100 = int_rectangle(f2, subdiv_reg(0, 1, 100))
    rec1000 = int_rectangle(f2, subdiv_reg(0, 1, 1000))
    rec10000 = int_rectangle(f2, subdiv_reg(0, 1, 10000))
    print("Integral of f2(x) from 0 to 1 using rectangles100: ", rec100)
    print("Integral of f2(x) from 0 to 1 using rectangles1000: ", rec1000)
    print("Integral of f2(x) from 0 to 1 using rectangles10000: ", rec10000)

    trap100 = int_trapez(f2, subdiv_reg(0, 1, 100))
    trap1000 = int_trapez(f2, subdiv_reg(0, 1, 1000))
    trap10000 = int_trapez(f2, subdiv_reg(0, 1, 10000))
    print("Integral of f2(x) from 0 to 1 using trapez100: ", trap100)
    print("Integral of f2(x) from 0 to 1 using trapez1000: ", trap1000)
    print("Integral of f2(x) from 0 to 1 using trapez10000: ", trap10000)

if (inputVal == 2):
    vl = periode(math.pi / 4)
    print("T(π/2) = ", vl)

if(inputVal == 3):
    trace_periode()

