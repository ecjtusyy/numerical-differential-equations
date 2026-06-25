# 使用有限体积法来求burgers方程

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 参数设置
# =========================
L = 1.0          # 空间区间长度 [0, L]
N = 200          # 控制体数量
T = 0.2          # 最终时间
CFL = 0.5        # CFL 数

dx = L / N       # 空间步长

# 单元中心
x = (np.arange(N) + 0.5) * dx


# =========================
# 2. 初始条件
# =========================
def initial_condition(x):
    return 0.5 + np.sin(2 * np.pi * x)


u = initial_condition(x)
u0 = u.copy()


# =========================
# 3. Burgers 方程通量
# f(u) = u^2 / 2
# =========================
def flux(u):
    return 0.5 * u ** 2


# =========================
# 4. Rusanov 数值通量
# F_{i+1/2}
# =========================
def numerical_flux(u_left, u_right):
    f_left = flux(u_left)
    f_right = flux(u_right)

    # Burgers 方程中 f'(u)=u，所以最大波速取 max|u|
    alpha = np.maximum(np.abs(u_left), np.abs(u_right))

    return 0.5 * (f_left + f_right) - 0.5 * alpha * (u_right - u_left)


# =========================
# 5. 有限体积时间推进
# =========================
t = 0.0
step = 0

while t < T:
    # 最大波速，决定时间步长
    max_speed = np.max(np.abs(u))

    if max_speed < 1e-12:
        dt = 1e-4
    else:
        dt = CFL * dx / max_speed

    # 最后一步不要超过 T
    if t + dt > T:
        dt = T - t

    # 周期边界：右邻居
    u_right = np.roll(u, -1)

    # 计算右边界通量 F_{i+1/2}
    F_iphalf = numerical_flux(u, u_right)

    # 左边界通量 F_{i-1/2}
    # 它就是前一个单元的右边界通量
    F_imhalf = np.roll(F_iphalf, 1)

    # 有限体积更新公式
    u = u - dt / dx * (F_iphalf - F_imhalf)

    t += dt
    step += 1


# =========================
# 6. 输出结果
# =========================
print("计算结束")
print("控制体数量 N =", N)
print("空间步长 dx =", dx)
print("最终时间 t =", t)
print("时间步数 step =", step)
print("u 最小值 =", np.min(u))
print("u 最大值 =", np.max(u))


# =========================
# 7. 作图
# =========================
plt.figure(figsize=(8, 5))
plt.plot(x, u0, label="初始解")
plt.plot(x, u, label="数值解")
plt.xlabel("x")
plt.ylabel("u")
plt.title("有限体积法求解 Burgers 方程（周期边界）")
plt.legend()
plt.grid(True)
plt.show()