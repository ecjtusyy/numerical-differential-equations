# 使用最一般的差分法来模拟burgers方程

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 参数设置
# =========================
L = 1.0
N = 200
T = 0.2
CFL = 0.5

dx = L / N
x = np.arange(N) * dx


# =========================
# 2. 初始条件
# =========================
def initial_condition(x):
    return 0.5 + np.sin(2 * np.pi * x)


u = initial_condition(x)
u0 = u.copy()


# =========================
# 3. Burgers 方程通量
# f(u)=u^2/2
# =========================
def flux(u):
    return 0.5 * u ** 2


# =========================
# 4. 时间推进
# =========================
t = 0.0
step = 0

while t < T:
    max_speed = np.max(np.abs(u))

    if max_speed < 1e-12:
        dt = 1e-4
    else:
        dt = CFL * dx / max_speed

    if t + dt > T:
        dt = T - t

    # 周期边界
    u_right = np.roll(u, -1)
    u_left = np.roll(u, 1)

    # 最普通的中心差分格式
    u_new = u - dt / (2 * dx) * (flux(u_right) - flux(u_left))

    u = u_new

    t += dt
    step += 1


# =========================
# 5. 输出结果
# =========================
print("计算结束")
print("方法 = 前向时间 + 中心空间差分")
print("控制点数量 N =", N)
print("空间步长 dx =", dx)
print("最终时间 t =", t)
print("时间步数 step =", step)
print("u 最小值 =", np.min(u))
print("u 最大值 =", np.max(u))

mass0 = np.sum(u0) * dx
mass = np.sum(u) * dx

print("初始总量 =", mass0)
print("最终总量 =", mass)
print("守恒误差 =", abs(mass - mass0))


# =========================
# 6. 作图
# =========================
plt.figure(figsize=(8, 5))
plt.plot(x, u0, label="初始解")
plt.plot(x, u, label="普通中心差分数值解")
plt.xlabel("x")
plt.ylabel("u")
plt.title("普通中心差分求解 Burgers 方程")
plt.legend()
plt.grid(True)
plt.show()