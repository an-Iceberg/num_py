import sympy as sp

sp.init_printing()

# Aufgabe 5.3 1. aus dem Skript

x1, x2, x3 = sp.symbols("x1 x2 x3")

f1 = sp.sin(x2 + 2 * x3)
f2 = sp.cos(2 * x1 + x2)
f = sp.Matrix([f1, f2])
x = sp.Matrix([x1, x2, x3])
x0 = sp.Matrix([sp.pi / 4, 0, sp.pi])

Df = f.jacobian(x)
print("Df(x₁,x₂) =")
sp.pprint(Df)
print()

print("f(x⁽⁰⁾) =")
sp.pprint(f.subs([(x1, x0[0]), (x2, x0[1]), (x3, x0[2])]))
print()

print("Df(x⁽⁰⁾) =")
sp.pprint(Df.subs([(x1, x0[0]), (x2, x0[1]), (x3, x0[2])]))
print()

print("(x - x⁽⁰⁾) =")
sp.pprint(x - x0)
print()

# g(x) = f(x⁽⁰⁾) + Df(x⁽⁰⁾)·(x - x⁽⁰⁾)
g = f.subs([(x1, x0[0]), (x2, x0[1]), (x3, x0[2])]) + Df.subs(
    [(x1, x0[0]), (x2, x0[1]), (x3, x0[2])]
) * (x - x0)
print("g(x) = f(x⁽⁰⁾) + Df(x⁽⁰⁾)·(x - x⁽⁰⁾) =")
sp.pprint(g)
