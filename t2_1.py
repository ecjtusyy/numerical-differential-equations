import numpy as np
import matplotlib.pyplot as plt
import os


"""
第二题第（1）问：五点菱形差分格式

原方程：
-(u_xx + u_yy) = f(x,y)

其中：
f(x,y) = (4y^2 - 2x^2) / (x^2 + 2y^2)^2

网格：
x_i = 1 + ih,  i = 0,1,...,m
y_j = jk,      j = 0,1,...,n

h = 1 / m
k = 3 / n

记：
U[i,j] ≈ u(x_i, y_j)

中心差分：
u_xx(x_i,y_j) ≈ (U[i-1,j] - 2U[i,j] + U[i+1,j]) / h^2
u_yy(x_i,y_j) ≈ (U[i,j-1] - 2U[i,j] + U[i,j+1]) / k^2

代入并整理得：
(2/h^2 + 2/k^2)U[i,j]
- (1/h^2)(U[i-1,j] + U[i+1,j])
- (1/k^2)(U[i,j-1] + U[i,j+1])
= f(x_i,y_j)
"""


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def f(x, y):
    return (4 * y**2 - 2 * x**2) / (x**2 + 2 * y**2)**2


def u_exact(x, y):
    return np.log(x**2 + 2 * y**2)


def get_omega(m, n, h, k):
    ax = 1 / h**2
    ay = 1 / k**2
    rho = (ax * np.cos(np.pi / m) + ay * np.cos(np.pi / n)) / (ax + ay)
    return 2 / (1 + np.sqrt(1 - rho**2))


def solve(m, n, tol=0.5e-10, max_iter=200000):
    h = 1 / m
    k = 3 / n

    x = np.linspace(1, 2, m + 1)
    y = np.linspace(0, 3, n + 1)

    U = np.zeros((m + 1, n + 1))

    # 边界值
    U[0, :] = np.log(1 + 2 * y**2)
    U[m, :] = np.log(4 + 2 * y**2)
    U[:, 0] = np.log(x**2)
    U[:, n] = np.log(x**2 + 18)

    ax = 1 / h**2
    ay = 1 / k**2
    center = 2 * ax + 2 * ay

    omega = get_omega(m, n, h, k)

    it = 0
    err = 1.0

    while it < max_iter and err > tol:
        err = 0.0
        it += 1

        for i in range(1, m):
            for j in range(1, n):
                old = U[i, j]

                gs = (
                    ax * (U[i - 1, j] + U[i + 1, j])
                    + ay * (U[i, j - 1] + U[i, j + 1])
                    + f(x[i], y[j])
                ) / center

                U[i, j] = old + omega * (gs - old)

                diff = abs(U[i, j] - old)
                if diff > err:
                    err = diff

    return x, y, U, it, err, omega


def print_points(m, n):
    x, y, U, it, iter_err, omega = solve(m, n)

    h = x[1] - x[0]
    k = y[1] - y[0]

    print(f"\n第二题第3问：m = {m}, n = {n}")
    print(f"SOR迭代次数：{it}")
    print(f"松弛因子 omega = {omega:.6f}")
    print(f"最后迭代误差 = {iter_err:.3e}")
    print("x        y        数值解             精确解             绝对误差")

    for xx in [1.25, 1.75]:
        for s in range(1, 6):
            yy = 0.5 * s

            i = int(round((xx - 1) / h))
            j = int(round(yy / k))

            num = U[i, j]
            exact = u_exact(xx, yy)
            err = abs(num - exact)

            print(f"{xx:<8.2f} {yy:<8.2f} {num:<18.12f} {exact:<18.12f} {err:.3e}")


def draw_figures():
    os.makedirs("figs", exist_ok=True)

    x1, y1, U1, it1, iter_err1, omega1 = solve(20, 30)
    x2, y2, U2, it2, iter_err2, omega2 = solve(40, 60)

    X1, Y1 = np.meshgrid(x1, y1, indexing="ij")
    X2, Y2 = np.meshgrid(x2, y2, indexing="ij")

    Ue1 = u_exact(X1, Y1)
    Ue2 = u_exact(X2, Y2)

    E1 = np.abs(U1 - Ue1)
    E2 = np.abs(U2 - Ue2)

    # 图1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    u_min = min(Ue2.min(), U1.min(), U2.min())
    u_max = max(Ue2.max(), U1.max(), U2.max())
    levels = np.linspace(u_min, u_max, 35)

    c0 = axes[0].contourf(X2, Y2, Ue2, levels=levels)
    axes[0].contour(X2, Y2, Ue2, levels=levels, linewidths=0.4)
    axes[0].set_title("精确解")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(c0, ax=axes[0])

    c1 = axes[1].contourf(X1, Y1, U1, levels=levels)
    axes[1].contour(X1, Y1, U1, levels=levels, linewidths=0.4)
    axes[1].set_title("数值解 m=20, n=30")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(c1, ax=axes[1])

    c2 = axes[2].contourf(X2, Y2, U2, levels=levels)
    axes[2].contour(X2, Y2, U2, levels=levels, linewidths=0.4)
    axes[2].set_title("数值解 m=40, n=60")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(c2, ax=axes[2])

    plt.suptitle("第二题第4问：精确解与两种数值解对比")
    plt.tight_layout()
    plt.savefig("第二题使用迭代第4问_图1_解函数对比图.png", dpi=300)
    plt.close()

    # 图2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    e_max = max(E1.max(), E2.max())
    levels_err = np.linspace(0, e_max, 35)

    c1 = axes[0].contourf(X1, Y1, E1, levels=levels_err)
    axes[0].set_title(f"m=20, n=30，最大误差={E1.max():.3e}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(c1, ax=axes[0])

    c2 = axes[1].contourf(X2, Y2, E2, levels=levels_err)
    axes[1].set_title(f"m=40, n=60，最大误差={E2.max():.3e}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(c2, ax=axes[1])

    plt.suptitle("第二题第4问：误差分布图")
    plt.tight_layout()
    plt.savefig("第二题使用迭代第4问_图2_误差分布图.png", dpi=300)
    plt.close()

    # 图3
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x_dense = np.linspace(1, 2, 300)

    for ax, yy in zip(axes, [0.5, 1.5, 2.5]):
        j1 = np.argmin(np.abs(y1 - yy))
        j2 = np.argmin(np.abs(y2 - yy))

        ax.plot(x_dense, u_exact(x_dense, yy), label="精确解")
        ax.plot(x1, U1[:, j1], "o", markersize=3, label="m=20, n=30")
        ax.plot(x2, U2[:, j2], "--", label="m=40, n=60")

        ax.set_title(f"y = {yy}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(True)
        ax.legend()

    plt.suptitle("第二题第4问：固定 y 截面上的解函数对比")
    plt.tight_layout()
    plt.savefig("第二题使用迭代第4问_图3_截面曲线图.png", dpi=300)
    plt.close()

    print("\n第二题第4问：三张图已保存到 figs 文件夹")
    print("第二题第4问_图1_解函数对比图.png")
    print("第二题第4问_图2_误差分布图.png")
    print("第二题第4问_图3_截面曲线图.png")


print_points(20, 30)
print_points(40, 60)

draw_figures()