# dfprp

GPU Fermat PRP test (base 3, the same residue Prime95 reports for "PRP-3, residue-type 1") for N = 2^n − c,
in OpenCL; dfprp = double-float PRP. Open `dfprp.slnx` in Visual Studio 2026, or run `build.bat` (Release by default).

## How it works

* Weighted real FFT: digit j has weight 2^(⌈nj/L⌉ − nj/L) · c^(j/L), which folds 2^n ≡ c into the transform,
  so a plain cyclic convolution of length L gives x² mod (2^n − c) with no zero padding (same idea as Prime95).
* Every floating-point value is a double-float (an unevaluated sum of two FP32 numbers, ~48-bit mantissa).
  Consumer GPUs such as the T4 and the Radeon 780M run FP64 at 1/32 of FP32, so this is the fastest variant there.
  The lower precision means ~12.2 bits per word, so n ≈ 4M uses L = 320K (5·2^16).
* Four-step FFT, W = 512 lines × H columns, three kernels per squaring (`fwdW`, `tail`, `invW`), two-level lazy
  carries, Gerbicz error check every B² iterations (B = 1000), checkpoint file `dfprp_<n>_<c>.ckpt`.

`dfprp.exe` loads `dfprp.cl` from its own folder at run time; the project copies it to `bin\<Configuration>\`.

## Running

```
dfprp n [c=25] [-iters k] [-prof] [-B gerbicz_block] [-nogec] [-fresh] [-report k] [-inject it] [-maxbits b] [-dev i]
```

* No `-iters`: full PRP test with Gerbicz check, resumable from the checkpoint.
* `-iters k -report k -prof`: benchmark k squarings and print per-kernel times.
* `check.py n [c] [iters]` prints the expected RES64 with Python (small n only).
* Debug's command line is a self-test: `20011 -nogec` must print RES64 `D6077DAFC085AB4C`.
* Measure speed with the Release build only, and only while Prime95 is stopped.

## Numbers on the Radeon 780M (n = 4,000,037)

0.397 ms per iteration, about 26.5 min per 4M-bit number, round-off error 0.03. Earlier variants, now removed
(still in git history before this commit): FP64 0.452 ms (full n = 3678495 run matched Prime95's res64 and
res2048), integer NTT over three primes 0.51 ms. After 2000 iterations at n = 4,000,037 all three gave RES64
146B8A66141B5E63.
