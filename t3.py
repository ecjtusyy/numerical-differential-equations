import numpy as np
import matplotlib.pyplot as plt


def u_exact(x, t):
    return t * np.sin(x)


def f(x, t):
    return (t + 1) * np.sin(x)


# 第1问：向前欧拉格式
# u_i^{j+1}=r*u_{i-1}^j+(1-2r)*u_i^j+r*u_{i+1}^j+tau*(t_j+1)*sin(x_i)
# 其中 r=tau/h^2，边界为 u(0,t)=0，u(1,t)=t*sin(1)
def forward_euler(h, tau):
    m = int(round(1 / h))
    n = int(round(1 / tau))

    x = np.linspace(0, 1, m + 1)
    t = np.linspace(0, 1, n + 1)

    u = np.zeros((n + 1, m + 1))
    r = tau / h**2

    # 初始条件
    u[0, :] = 0

    # 边界条件
    u[:, 0] = 0
    u[:, -1] = t * np.sin(1)

    # 第2问：用向前欧拉格式计算内部点
    for j in range(n):
        for i in range(1, m):
            u[j + 1, i] = (
                r * u[j, i - 1]
                + (1 - 2 * r) * u[j, i]
                + r * u[j, i + 1]
                + tau * f(x[i], t[j])
            )

        # 每一层右边界随时间变化
        u[j + 1, 0] = 0
        u[j + 1, -1] = t[j + 1] * np.sin(1)

    return x, t, u, r


# 第3问：输出指定点 (0.5, 0.2i), i=1,2,3,4,5
def print_table(h, tau):
    x, t, u, r = forward_euler(h, tau)

    print()
    print(f"h = {h:.5f}, tau = {tau:.5f}, r = {r:.5f}")
    if r > 0.5:
        print("这一组 r > 0.5，结果可能不稳定")

    print("x        t        数值解             精确解             误差")

    ix = int(round(0.5 / h))

    for k in range(1, 6):
        t0 = 0.2 * k
        jt = int(round(t0 / tau))

        num = u[jt, ix]
        exact = u_exact(0.5, t0)
        err = abs(num - exact)

        print(f"{0.5:<8.2f} {t0:<8.2f} {num:<18.12f} {exact:<18.12f} {err:.3e}")


# 第4问：画精确解、数值解和误差图
def plot_result():
    h1, tau1 = 1 / 4, 1 / 100
    h2, tau2 = 1 / 8, 1 / 200

    x1, t1, u1, r1 = forward_euler(h1, tau1)
    x2, t2, u2, r2 = forward_euler(h2, tau2)

    T = 1.0

    exact1 = u_exact(x1, T)
    exact2 = u_exact(x2, T)

    num1 = u1[-1, :]
    num2 = u2[-1, :]

    err1 = np.abs(num1 - exact1)
    err2 = np.abs(num2 - exact2)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure()
    plt.plot(x2, exact2, label="精确解")
    plt.plot(x1, num1, "o-", label="h=1/4, tau=1/100")
    plt.plot(x2, num2, "s-", label="h=1/8, tau=1/200")
    plt.xlabel("x")
    plt.ylabel("u(x,1)")
    plt.title("t=1 时精确解与数值解对比")
    plt.legend()
    plt.grid(True)
    plt.savefig("第三题_精确解和数值解对比.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(x1, err1, "o-", label="h=1/4, tau=1/100")
    plt.plot(x2, err2, "s-", label="h=1/8, tau=1/200")
    plt.xlabel("x")
    plt.ylabel("误差")
    plt.title("t=1 时误差图")
    plt.legend()
    plt.grid(True)
    plt.savefig("第三题_误差图.png", dpi=300)
    plt.show()


# 四组步长
steps = [
    (1 / 4, 1 / 100),
    (1 / 4, 1 / 200),
    (1 / 8, 1 / 100),
    (1 / 8, 1 / 200),
]

for h, tau in steps:
    print_table(h, tau)

plot_result()