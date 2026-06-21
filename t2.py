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

代入：
-[(U[i-1,j] - 2U[i,j] + U[i+1,j]) / h^2
 +(U[i,j-1] - 2U[i,j] + U[i,j+1]) / k^2]
= f(x_i,y_j)

整理得：
(2/h^2 + 2/k^2)U[i,j]
- (1/h^2)(U[i-1,j] + U[i+1,j])
- (1/k^2)(U[i,j-1] + U[i,j+1])
= f(x_i,y_j)

写成线性方程组：
A x = b

其中 x 是所有内点未知量 U[i,j] 排成的一维向量。
"""


# 设置中文标题，避免图片中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def f(x, y):
    return (4 * y**2 - 2 * x**2) / (x**2 + 2 * y**2)**2


def u_exact(x, y):
    return np.log(x**2 + 2 * y**2)


def solve(m, n):
    # 网格步长
    h = 1 / m
    k = 3 / n

    # 网格点
    x = np.linspace(1, 2, m + 1)
    y = np.linspace(0, 3, n + 1)

    # 数值解数组，边界先放进去
    U = np.zeros((m + 1, n + 1))

    # 边界条件
    U[0, :] = np.log(1 + 2 * y**2)
    U[m, :] = np.log(4 + 2 * y**2)
    U[:, 0] = np.log(x**2)
    U[:, n] = np.log(x**2 + 18)

    # 五点格式系数
    ax = 1 / h**2
    ay = 1 / k**2
    center = 2 * ax + 2 * ay

    # 内点未知量个数
    N = (m - 1) * (n - 1)

    # 构造线性方程组 A sol = b
    A = np.zeros((N, N))
    b = np.zeros(N)

    # 把二维内点编号 (i,j) 变成一维编号 p
    # 未知量顺序：
    # U[1,1], U[1,2], ..., U[1,n-1],
    # U[2,1], U[2,2], ..., U[2,n-1],
    # ...
    # U[m-1,1], ..., U[m-1,n-1]
    def idx(i, j):
        return (i - 1) * (n - 1) + (j - 1)

    # 构造 A 和 b
    for i in range(1, m):
        for j in range(1, n):
            p = idx(i, j)

            # 当前内点方程：
            # center * U[i,j]
            # - ax * U[i-1,j]
            # - ax * U[i+1,j]
            # - ay * U[i,j-1]
            # - ay * U[i,j+1]
            # = f(x_i,y_j)

            A[p, p] = center
            b[p] = f(x[i], y[j])

            # 左邻点 U[i-1,j]
            if i - 1 >= 1:
                q = idx(i - 1, j)
                A[p, q] = -ax
            else:
                # 碰到左边界，移到右端项
                b[p] += ax * U[i - 1, j]

            # 右邻点 U[i+1,j]
            if i + 1 <= m - 1:
                q = idx(i + 1, j)
                A[p, q] = -ax
            else:
                # 碰到右边界，移到右端项
                b[p] += ax * U[i + 1, j]

            # 下邻点 U[i,j-1]
            if j - 1 >= 1:
                q = idx(i, j - 1)
                A[p, q] = -ay
            else:
                # 碰到下边界，移到右端项
                b[p] += ay * U[i, j - 1]

            # 上邻点 U[i,j+1]
            if j + 1 <= n - 1:
                q = idx(i, j + 1)
                A[p, q] = -ay
            else:
                # 碰到上边界，移到右端项
                b[p] += ay * U[i, j + 1]

    # 直接解线性方程组
    sol = np.linalg.solve(A, b)

    # 把一维解向量放回二维网格
    for i in range(1, m):
        for j in range(1, n):
            p = idx(i, j)
            U[i, j] = sol[p]

    return x, y, U


def print_points(m, n):
    x, y, U = solve(m, n)

    h = x[1] - x[0]
    k = y[1] - y[0]

    print(f"\n第二题第3问：m = {m}, n = {n}")
    print("求解方法：五点菱形格式离散后，构造线性方程组 A x = b，并用 np.linalg.solve 求解")
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


def draw_exact():
    # 第二题第4问：精确解图只需要画一张
    x = np.linspace(1, 2, 101)
    y = np.linspace(0, 3, 101)

    X, Y = np.meshgrid(x, y, indexing="ij")
    Ue = u_exact(X, Y)

    os.makedirs("figs", exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Ue)
    ax.set_title("第二题第4问 精确解图")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    plt.savefig("figs/第二题第4问_精确解图.png", dpi=300)
    plt.close()

    print("第二题第4问：精确解图已保存到 figs 文件夹")


def draw_num_err(m, n):
    x, y, U = solve(m, n)

    X, Y = np.meshgrid(x, y, indexing="ij")
    Ue = u_exact(X, Y)
    Err = np.abs(U - Ue)

    os.makedirs("figs", exist_ok=True)

    # 第二题第4问：数值解图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, U)
    ax.set_title(f"第二题第4问 数值解图 m={m}, n={n}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("U")
    plt.savefig(f"figs/第二题第4问_数值解图_m{m}_n{n}.png", dpi=300)
    plt.close()

    # 第二题第4问：误差图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Err)
    ax.set_title(f"第二题第4问 误差图 m={m}, n={n}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("|U-u|")
    plt.savefig(f"figs/第二题第4问_误差图_m{m}_n{n}.png", dpi=300)
    plt.close()

    print(f"第二题第4问：m={m}, n={n} 的数值解图和误差图已保存到 figs 文件夹")


# 第二题第3问：输出两种剖分下的 10 个节点结果
print_points(20, 30)
print_points(40, 60)

# 第二题第4问：作图
draw_exact()
draw_num_err(20, 30)
draw_num_err(40, 60)