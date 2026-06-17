import sympy as sp

class to_standard:
    def __init__(self,A,B,C):
        # 初始化输入值
        # A是X^2
        # B是XY
        # C是Y^2
        
        self.A=A
        self.B=B
        self.C=C

        # 初始化二次型矩阵
        self.M=sp.Matrix([
            [A,B],
            [B,C]
        ])

    def get_type(self):
        D = sp.simplify(self.B**2 - self.A * self.C)

        if D == 0:
            return D, "抛物型"
        elif D>0:
            return D, "双曲型"
        elif D<0:
            return D, "椭圆型"

    def orthogonal_mat(self):
        # 1.求特征值和特征向量
        data = self.M.eigenvects()

        values = []
        vectors = []

        for item in data:
            value = item[0]       # 特征值
            vec_list = item[2]    # 特征向量

            for v in vec_list:
                v = sp.Matrix(v)

                # 2.单位化特征向量
                length = sp.sqrt(v[0]**2 + v[1]**2)
                unit_v = sp.simplify(v / length)

                values.append(value)
                vectors.append(unit_v)

        # 3.组成正交矩阵Q
        Q = sp.Matrix.hstack(vectors[0], vectors[1])

        # 4.正交变换 Q^T M Q
        new_M = sp.simplify(Q.T * self.M * Q)

        return values, Q, new_M

    def solve(self):
        D, pde_type = self.get_type()
        values, Q, new_M = self.orthogonal_mat()

        print("类型：", pde_type)
        print("正交变换后矩阵：", new_M)
        print(f"标准型：({new_M[0,0]})U_xixi + ({new_M[1,1]})U_etaeta = 0")

x, y = sp.symbols("x y", positive=True)

pde = to_standard(x**2, x*y, y**2)

pde.solve()
