import numpy as np
import matplotlib.pyplot as plt


# 精确解
def u_exact(x, t):
    return np.exp(x + 0.1 * t)


# 初始位移 u(x,0)
def phi(x):
    return np.exp(x)


# 初始速度 u_t(x,0)
def psi(x):
    return 0.1 * np.exp(x)


# 左右边界
def left_bc(t):
    return np.exp(0.1 * t)


def right_bc(t):
    return np.exp(10 + 0.1 * t)


# 第1问：差分递推公式
# u_i^{j+1}=2(1-r^2)u_i^j-u_i^{j-1}+r^2(u_{i+1}^j+u_{i-1}^j)
# r=c*tau/h，第一层用初始速度单独算
def wave_solve(h=0.05, tau=0.005, c=0.1):
    x = np.arange(0, 10 + h, h)
    t = np.arange(0, 10 + tau, tau)

    m = len(x) - 1
    n = len(t) - 1

    u = np.zeros((n + 1, m + 1))
    r = c * tau / h

    if r > 1:
        print("注意：r > 1，显式格式不稳定")

    # 初始条件
    u[0, :] = phi(x)

    # 边界条件
    u[:, 0] = left_bc(t)
    u[:, -1] = right_bc(t)

    # 由初始速度计算第一层
    for i in range(1, m):
        u[1, i] = (
            u[0, i]
            + tau * psi(x[i])
            + 0.5 * r**2 * (u[0, i + 1] - 2 * u[0, i] + u[0, i - 1])
        )

    u[1, 0] = left_bc(t[1])
    u[1, -1] = right_bc(t[1])

    # 第2问：递推计算后面的时间层
    for j in range(1, n):
        for i in range(1, m):
            u[j + 1, i] = (
                2 * (1 - r**2) * u[j, i]
                - u[j - 1, i]
                + r**2 * (u[j, i + 1] + u[j, i - 1])
            )

        u[j + 1, 0] = left_bc(t[j + 1])
        u[j + 1, -1] = right_bc(t[j + 1])

    return x, t, u, r


# 方便取指定时刻或指定位置的下标
def near_index(arr, value):
    return int(np.argmin(np.abs(arr - value)))


# 第2问(a)：画 x-t-u 三维曲面
def plot_surface(x, t, u):
    X, T = np.meshgrid(x, t)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, T, u, cmap="viridis")

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    ax.set_title("第四题：x-t-u 数值解曲面")

    plt.savefig("第四题_x-t-u三维曲面.png", dpi=300)
    plt.show()


# 第2问(b)：画 t=0,0.5,1 时的 x-u 曲线，并和解析解比较
def plot_x_u(x, t, u):
    times = [0, 0.5, 1]

    plt.figure()

    for tt in times:
        j = near_index(t, tt)
        plt.plot(x, u[j, :], "o-", markersize=3, label=f"数值解 t={tt}")
        plt.plot(x, u_exact(x, tt), "--", label=f"解析解 t={tt}")

    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("第四题：不同时刻的 x-u 曲线")
    plt.legend()
    plt.grid(True)

    plt.savefig("第四题_x-u曲线比较.png", dpi=300)
    plt.show()


# 第2问(c)：画 x=0.5 处的 t-u 曲线，并和解析解比较
def plot_t_u(x, t, u):
    x0 = 0.5
    i = near_index(x, x0)

    num = u[:, i]
    exact = u_exact(x0, t)

    plt.figure()
    plt.plot(t, num, label="数值解")
    plt.plot(t, exact, "--", label="解析解")

    plt.xlabel("t")
    plt.ylabel("u")
    plt.title("第四题：x=0.5 处的 t-u 曲线")
    plt.legend()
    plt.grid(True)

    plt.savefig("第四题_t-u曲线比较.png", dpi=300)
    plt.show()


# 误差图，作业里可以顺便放上
def plot_error(x, t, u):
    X, T = np.meshgrid(x, t)
    exact = u_exact(X, T)
    err = np.abs(u - exact)

    plt.figure()
    plt.imshow(
        err,
        extent=[x[0], x[-1], t[-1], t[0]],
        aspect="auto"
    )
    plt.colorbar(label="误差")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("第四题：误差分布图")

    plt.savefig("第四题_误差分布图.png", dpi=300)
    plt.show()


# 主程序
h = 0.05
tau = 0.005
c = 0.1       

x, t, u, r = wave_solve(h, tau, c)

print(f"h = {h}")
print(f"tau = {tau}")
print(f"r = c*tau/h = {r}")
print("计算完成")

plot_surface(x, t, u)
plot_x_u(x, t, u)
plot_t_u(x, t, u)
plot_error(x, t, u)