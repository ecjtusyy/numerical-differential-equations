# 使用正交矩阵不符合偏微分方程的化标准型
# 经过探索了解到要使用雅可比行列式作为合同变换才是符合要求的
import sympy as sp

class to_standard:
    def __init__(self,A,B,C,x,y):
        # 初始化输入值
        self.A=A
        self.B=B
        self.C=C
        self.x=x
        self.y=y

        # 初始化二次型矩阵
        self.M=sp.Matrix([
            [A,B],
            [B,C]
        ])

    def is_zero(self,expr):
        # 判断是否为0
        return sp.simplify(expr).equals(0) is True

    def get_type(self):
        D = sp.simplify(self.B**2 - self.A * self.C)

        if self.is_zero(D):
            return D, "抛物型"
        elif D.is_positive:
            return D, "双曲型"
        elif D.is_negative:
            return D, "椭圆型"
        else:
            return D, "类型无法直接判断"

    def get_var_from_vec(self,vec,name):
        x = self.x
        y = self.y

        # 特征向量 v=(p,q)
        p = sp.simplify(vec[0])
        q = sp.simplify(vec[1])

        # 令新变量的梯度平行于特征向量
        # 所以沿着垂直方向(-q,p)导数为0
        f = sp.Function(name)
        eq = sp.Eq(-q*sp.diff(f(x,y),x) + p*sp.diff(f(x,y),y), 0)

        sol = sp.pdsolve(eq)

        # pdsolve一般得到 f(x,y)=F(...)
        var = sol.rhs.args[0]

        return sp.simplify(var)

    def get_change(self):
        # 用矩阵求特征值和特征向量
        data = self.M.eigenvects()

        vars = []

        for item in data:
            vec = sp.Matrix(item[2][0])
            var = self.get_var_from_vec(vec, "f")
            vars.append(var)

        xi = vars[0]
        eta = vars[1]

        return xi, eta

    def get_new_matrix(self,xi,eta):
        x = self.x
        y = self.y

        # 求雅可比矩阵J
        J = sp.Matrix([
            [sp.diff(xi,x), sp.diff(xi,y)],
            [sp.diff(eta,x), sp.diff(eta,y)]
        ])

        # 用雅可比矩阵做合同变换
        new_M = J * self.M * J.T
        new_M = new_M.applyfunc(sp.factor)

        det_J = sp.factor(sp.simplify(J.det()))

        return J, det_J, new_M

    def standard_text(self,new_M):
        # 根据矩阵自动写标准型
        a = sp.factor(new_M[0,0])
        b = sp.factor(new_M[0,1])
        c = sp.factor(new_M[1,1])

        terms = []

        if not self.is_zero(a):
            terms.append(f"({a})U_xixi")

        if not self.is_zero(b):
            terms.append(f"({2*b})U_xieta")

        if not self.is_zero(c):
            terms.append(f"({c})U_etaeta")

        if len(terms) == 0:
            return "0 = 0"

        return " + ".join(terms) + " = 0"

    def solve(self):
        D, pde_type = self.get_type()

        xi, eta = self.get_change()
        J, det_J, new_M = self.get_new_matrix(xi,eta)

        print("类型：", pde_type)
        print("换元：xi =", xi)
        print("换元：eta =", eta)
        print("detJ：", det_J)
        print("标准型：", self.standard_text(new_M))


x, y = sp.symbols("x y", positive=True)

pde = to_standard(x**2, x*y, y**2, x, y)

pde.solve()