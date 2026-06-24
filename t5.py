# 使用mpm来计算burger方程

import numpy as np
import matplotlib.pyplot as plt


# 一维线性形函数，周期边界
def shape_linear_periodic(xp, dx, nx):
    s = xp / dx
    i0 = int(np.floor(s)) % nx
    r = s - np.floor(s)

    i1 = (i0 + 1) % nx

    # 形函数
    N0 = 1.0 - r
    N1 = r

    # 形函数对 x 的导数
    dN0 = -1.0 / dx
    dN1 = 1.0 / dx

    return (i0, i1), (N0, N1), (dN0, dN1)


def run_mpm_burgers(nx=100, ppc=2, t_end=0.05, cfl=0.2):
    """
    一维无黏 Burgers 方程：
        u_t + (u^2 / 2)_x = 0

    周期边界，所以分部积分后的边界项为 0。
    """

    # =========================
    # 0. 网格和粒子初始化
    # =========================

    L = 1.0
    dx = L / nx

    npart = nx * ppc

    # 粒子位置
    xp = (np.arange(npart) + 0.5) * L / npart

    # 每个粒子代表的长度，也就是一维里的 Vp
    Vp = np.full(npart, L / npart)

    # 粒子携带的 Burgers 变量 u_p
    up = 0.5 + 0.5 * np.sin(2 * np.pi * xp)

    t = 0.0
    step = 0

    while t < t_end:
        # CFL 时间步
        umax = max(np.max(np.abs(up)), 1e-12)
        dt = cfl * dx / umax

        if t + dt > t_end:
            dt = t_end - t

        # =========================
        # 1. 清空网格量
        # =========================

        M = np.zeros(nx)       # 节点质量/节点权重 M_i
        U_num = np.zeros(nx)   # 用来算节点 U_i 的分子
        R = np.zeros(nx)       # 弱形式右端 R_i

        # =========================
        # 2. 粒子到网格：计算 M_i 和 U_i
        # =========================
        #
        # M_i = sum_p V_p N_i(x_p)
        #
        # U_i = sum_p V_p N_i(x_p) u_p / M_i
        #

        for p in range(npart):
            ids, Ns, _ = shape_linear_periodic(xp[p], dx, nx)

            for a in range(2):
                i = ids[a]
                N = Ns[a]

                M[i] += Vp[p] * N
                U_num[i] += Vp[p] * N * up[p]

        U = np.zeros(nx)
        mask = M > 1e-14
        U[mask] = U_num[mask] / M[mask]

        # =========================
        # 3. 网格中积分计算：弱形式右端
        # =========================
        #
        # Burgers 方程：
        #     u_t + (u^2 / 2)_x = 0
        #
        # 乘 N_i，积分，分部积分后：
        #     ∫ N_i u_t dx = ∫ N_i,x (u^2 / 2) dx
        #
        # MPM 用粒子求和近似：
        #     R_i = sum_p V_p N_i,x(x_p) (u_p^2 / 2)
        #

        for p in range(npart):
            ids, _, dNs = shape_linear_periodic(xp[p], dx, nx)

            flux = 0.5 * up[p] ** 2

            for a in range(2):
                i = ids[a]
                dN = dNs[a]

                R[i] += Vp[p] * dN * flux

        # =========================
        # 4. 处理第一个积分里的 t 导数
        # =========================
        #
        # 第一个积分：
        #     ∫ N_i u_t dx
        #
        # 近似成：
        #     M_i dU_i/dt
        #
        # 再用时间差分：
        #     M_i (U_i^{n+1} - U_i^n) / dt = R_i
        #
        # 所以：
        #     U_i^{n+1} = U_i^n + dt * R_i / M_i
        #

        U_new = U.copy()
        U_new[mask] = U[mask] + dt * R[mask] / M[mask]

        # =========================
        # 5. 网格回粒子：更新 u_p
        # =========================
        #
        # u_p^{n+1} = sum_i N_i(x_p) U_i^{n+1}
        #

        up_new = np.zeros_like(up)

        for p in range(npart):
            ids, Ns, _ = shape_linear_periodic(xp[p], dx, nx)

            val = 0.0
            for a in range(2):
                i = ids[a]
                N = Ns[a]
                val += N * U_new[i]

            up_new[p] = val

        up = up_new

        # =========================
        # 6. 粒子位置推进
        # =========================
        #
        # Burgers 方程的特征速度是 u
        # 所以可以用：
        #     x_p^{n+1} = x_p^n + dt * u_p
        #

        xp = (xp + dt * up) % L

        t += dt
        step += 1

    print("计算结束")
    print("步数 =", step)
    print("最终时间 t =", t)
    print("u 最小值 =", np.min(up))
    print("u 最大值 =", np.max(up))

    return xp, up


if __name__ == "__main__":
    xp, up = run_mpm_burgers(
        nx=100,
        ppc=2,
        t_end=0.05,
        cfl=0.2
    )

    order = np.argsort(xp)

    plt.figure()
    plt.plot(xp[order], up[order], marker="o", markersize=3)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("MPM 弱形式求解一维 Burgers 方程")
    plt.grid(True)
    plt.show()