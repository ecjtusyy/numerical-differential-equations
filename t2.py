import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return (4 * y**2 - 2 * x**2) / (x**2 + 2 * y**2)**2


def u_exact(x, y):
    return np.log(x**2 + 2 * y**2)


def solve(m, n, omega=1.85, tol=0.5e-10, max_iter=200000):
    # 初始化网格
    h = 1 / m
    k = 3 / n

    x = np.linspace(1, 2, m + 1)
    y = np.linspace(0, 3, n + 1)

    U = np.zeros((m + 1, n + 1))

    # 边界条件
    U[0, :] = np.log(1 + 2 * y**2)
    U[m, :] = np.log(4 + 2 * y**2)
    U[:, 0] = np.log(x**2)
    U[:, n] = np.log(x**2 + 18)

    # 右端项
    X, Y = np.meshgrid(x, y, indexing="ij")
    F = f(X, Y)

    ax = 1 / h**2
    ay = 1 / k**2
    center = 2 * ax + 2 * ay

    # SOR 迭代
    for it in range(1, max_iter + 1):
        max_change = 0

        for i in range(1, m):
            for j in range(1, n):
                old = U[i, j]

                gs = (F[i, j]
                      + ax * (U[i - 1, j] + U[i + 1, j])
                      + ay * (U[i, j - 1] + U[i, j + 1])) / center

                U[i, j] = (1 - omega) * old + omega * gs

                max_change = max(max_change, abs(U[i, j] - old))

        if max_change < tol:
            break

    return x, y, U, it, max_change


def print_points(m, n):
    x, y, U, it, max_change = solve(m, n)

    h = x[1] - x[0]
    k = y[1] - y[0]

    print(f"\nm = {m}, n = {n}")
    print(f"迭代次数 = {it}, 最大更新量 = {max_change:.3e}")
    print("x        y        数值解             精确解             误差")

    for xx in [1.25, 1.75]:
        for s in range(1, 6):
            yy = 0.5 * s

            i = int(round((xx - 1) / h))
            j = int(round(yy / k))

            num = U[i, j]
            exact = u_exact(xx, yy)
            err = abs(num - exact)

            print(f"{xx:<8.2f} {yy:<8.2f} {num:<18.12f} {exact:<18.12f} {err:.3e}")


import os

def draw_one(m, n):
    x, y, U, it, max_change = solve(m, n)

    X, Y = np.meshgrid(x, y, indexing="ij")
    Ue = u_exact(X, Y)
    Err = np.abs(U - Ue)

    os.makedirs("figs", exist_ok=True)

    # 精确解图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Ue)
    ax.set_title(f"第二题 精确解 m={m}, n={n}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    plt.savefig(f"figs/第二题_精确解_m{m}_n{n}.png", dpi=300)
    plt.close()

    # 数值解图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, U)
    ax.set_title(f"第二题 数值解 m={m}, n={n}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("U")
    plt.savefig(f"figs/第二题_数值解_m{m}_n{n}.png", dpi=300)
    plt.close()

    # 误差图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Err)
    ax.set_title(f"第二题 误差图 m={m}, n={n}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("|U-u|")
    plt.savefig(f"figs/第二题_误差图_m{m}_n{n}.png", dpi=300)
    plt.close()

    print(f"第二题 m={m}, n={n} 的三张图已经保存到 figs 文件夹")


# 输出 10 个指定节点
print_points(20, 30)
print_points(40, 60)

# 画图
draw_one(20, 30)
draw_one(40, 60)