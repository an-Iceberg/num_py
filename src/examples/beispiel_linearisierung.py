import sympy as sp

sp.init_printing(use_unicode=True)

# Beispiel 5.3 1. aus dem Skript

x1, x2 = sp.symbols("x1 x2")

f1 = x1**2 + x2 - 11
f2 = x1 + x2**2 - 7

f = sp.Matrix([f1, f2])
x = sp.Matrix([x1, x2])
x0 = sp.Matrix([1, 1])
Df = f.jacobian(x)

print("Df(x₁,x₂) =")
sp.pprint(Df)
print()

print("f(x⁽⁰⁾) =")
sp.pprint(f.subs([(x1, 1), (x2, 1)]))
print()

print("Df(x⁽⁰⁾) =")
sp.pprint(Df.subs([(x1, 1), (x2, 1)]))
print()

print("(x - x⁽⁰⁾) =")
sp.pprint(x - x0)
print()

# g(x) = f(x⁽⁰⁾) + Df(x⁽⁰⁾)·(x - x⁽⁰⁾)
g = f.subs([(x1, 1), (x2, 1)]) + (Df.subs([(x1, 1), (x2, 1)]) * (x - x0))
print("g(x) = f(x⁽⁰⁾) + Df(x⁽⁰⁾)·(x - x⁽⁰⁾) =")
sp.pprint(g)
