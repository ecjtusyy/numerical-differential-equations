# 使用有限元来模拟burgers方程

import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 参数设置
# =========================
L = 1.0          # 区间 [0,1]
N = 100          # 单元个数
T = 0.05         # 终止时间
dt = 0.0002      # 时间步长

h = L / N
x = np.linspace(0, L, N + 1)
steps = int(T / dt)

# =========================
# 2. 初始条件
# =========================
def initial_condition(x):
    return np.sin(np.pi * x)

U = initial_condition(x)

# 零边界条件
U[0] = 0.0
U[-1] = 0.0

U0 = U.copy()

# =========================
# 3. 组装质量矩阵 M
# =========================
M = np.zeros((N + 1, N + 1))

for e in range(N):
    nodes = [e, e + 1]

    # 一次有限元局部质量矩阵
    Me = h / 6.0 * np.array([
        [2.0, 1.0],
        [1.0, 2.0]
    ])

    for a in range(2):
        for b in range(2):
            M[nodes[a], nodes[b]] += Me[a, b]

# 只求内部节点，边界节点固定为 0
inner = np.arange(1, N)
M_inner = M[np.ix_(inner, inner)]


# =========================
# 4. 组装非线性项 C(U)
# =========================
def convection_vector(U):
    """
    计算 C_i(U) = ∫ phi_i' * (u_h^2 / 2) dx
    """
    C = np.zeros(N + 1)

    for e in range(N):
        UL = U[e]
        UR = U[e + 1]

        # 单元上 ∫ u_h^2 / 2 dx 的导数贡献
        value = (UL**2 + UL * UR + UR**2) / 6.0

        # 左节点 phi' = -1/h，右节点 phi' = 1/h
        C[e]     += -value
        C[e + 1] +=  value

    return C


# =========================
# 5. 时间推进
# =========================
for n in range(steps):
    C = convection_vector(U)

    rhs = dt * C[inner]

    # 解 M * dU = dt * C
    dU = np.linalg.solve(M_inner, rhs)

    U[inner] = U[inner] + dU

    # 保持零边界条件
    U[0] = 0.0
    U[-1] = 0.0


# =========================
# 6. 输出结果
# =========================
print("计算结束")
print("方程：u_t + (u^2 / 2)_x = 0")
print("边界条件：u(0,t)=0, u(1,t)=0")
print("单元个数 N =", N)
print("空间步长 h =", h)
print("时间步长 dt =", dt)
print("最终时间 T =", T)
print("u 最小值 =", np.min(U))
print("u 最大值 =", np.max(U))

print("\n部分节点数值解：")
for i in range(0, N + 1, 10):
    print(f"x = {x[i]:.2f}, u = {U[i]:.8f}")


# =========================
# 7. 作图
# =========================
plt.figure(figsize=(7, 4))
plt.plot(x, U0, "--", label="initial condition")
plt.plot(x, U, label="FEM solution")
plt.xlabel("x")
plt.ylabel("u")
plt.title("FEM for Burgers equation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("FEM_Burgers.png", dpi=300)
plt.show()