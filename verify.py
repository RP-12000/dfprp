import sys
lines = open(sys.argv[1]).read().split()
n, c, w, todo = (int(x) for x in lines[:4])
digits = [int(x) for x in lines[4:]]
N = (1 << n) - c
val = sum(d << (w * i) for i, d in enumerate(digits)) % N
E = N - 1
e = E >> (n - 1 - todo)
expect = pow(3, e, N)
print(f"n={n} iterations={todo}: GPU residue {'MATCHES' if val == expect else 'DIFFERS FROM'} Python pow()")
if todo == n - 1:
    print("full test:", "PRP (3^(N-1) == 1)" if val == 1 else f"composite, res64={val & ((1 << 64) - 1):016X}")
