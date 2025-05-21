import sympy as sp

sp.init_printing(use_unicode=True)

# Beispiel 5.3 2. aus dem Skript

x1, x2 = sp.symbols("x1 x2")

f = x1**2 + x2**2
x = sp.Matrix([x1, x2])
x0 = sp.Matrix([1, 2])
Df = sp.Matrix([sp.diff(f, x1), sp.diff(f, x2)])

print("Df(x₁,x₂) =")
sp.pprint(Df)
print()

print("f(x⁽⁰⁾) =")
sp.pprint(f.subs([(x1, 1), (x2, 2)]))
print()

print("nDf(x⁽⁰⁾) =")
sp.pprint(Df.subs([(x1, 1), (x2, 2)]))
print()

print("(x - x⁽⁰⁾) =")
sp.pprint(x - x0)
print()

# g(x) = f(x⁽⁰⁾) + Df(x⁽⁰⁾)·(x - x⁽⁰⁾)
g = sp.Matrix([f.subs([(x1, 1), (x2, 2)])]) + Df.subs([(x1, 1), (x2, 2)]).T * (x - x0)
print("g(x) = f(x⁽⁰⁾) + Df(x⁽⁰⁾)·(x - x⁽⁰⁾) =")
sp.pprint(g)
