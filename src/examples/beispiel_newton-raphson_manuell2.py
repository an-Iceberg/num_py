import sympy as sp

sp.init_printing(use_unicode=True)

x, y, c = sp.symbols("x y c")

f1 = c * x + y - 4
f2 = x**2 + y**2 - 9

f = sp.Matrix([f1, f2])
X = sp.Matrix([x, y])
Df = f.jacobian(X)

x0 = [0, 3]

print("Df =")
sp.pprint(Df)
print()

print("Df(x⁽⁰⁾) =")
sp.pprint(Df.subs([(x, x0[0]), (y, x0[1])]))
print()

print("Df(x⁽⁰⁾)⁻¹ =")
sp.pprint(Df.subs([(x, x0[0]), (y, x0[1])]).inv())
print()

print("-f(x⁽⁰⁾) =")
sp.pprint(-f.subs([(x, x0[0]), (y, x0[1])]))
print()

print("δ⁽⁰⁾ = Df(x⁽⁰⁾)⁻¹·-f(x⁽⁰⁾) =")
δ = Df.subs([(x, x0[0]), (y, x0[1])]).inv() * -f.subs([(x, x0[0]), (y, x0[1])])
sp.pprint(δ)
print()

print("x⁽¹⁾ = x⁽⁰⁾ + δ⁽⁰⁾ =")
sp.pprint(X.subs([(x, x0[0]), (y, x0[1])]) + δ)
print("=")
sp.pprint(sp.simplify(X.subs([(x, x0[0]), (y, x0[1])]) + δ))
