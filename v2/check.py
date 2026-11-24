import sys
n = int(sys.argv[1]); c = int(sys.argv[2]) if len(sys.argv) > 2 else 25
todo = int(sys.argv[3]) if len(sys.argv) > 3 else n - 1
N = (1 << n) - c
e = (N - 1) >> (n - 1 - todo)
r = pow(3, e, N)
print(f"RES64 {r & ((1<<64)-1):016X}")
