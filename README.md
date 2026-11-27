# dfprp

**All Purpose k·b^n+c PRP tester on GPU.** Version 1.0, developed on 2026.11.25 (the program prints this first).

The GPU program is `dfprp` (sources `hyprp.cpp` and `hyprp.cl`; FP32 + M31 hybrid FFT): a Fermat PRP test, base 3, with the same residue Prime95
reports for "PRP-3, residue-type 1". It runs through CUDA on NVIDIA GPUs (when built with `-DHYPRP_CUDA`) and
through OpenCL on every other GPU. Open `dfprp.slnx` in Visual Studio 2026, or run `build.bat` (Release by default;
the Windows build uses OpenCL).

**Supported so far:** k = 1, b = 2: N = 2^n − c with c > 0 (tested with c = 25; the FP32 + M31 mode also needs c to
be a square mod 2^31 − 1, which dfprp checks at the start), n up to 34,603,008. Other k, b and +c are not
implemented yet.

## How it works

* Weighted real FFT: digit j has weight 2^(⌈nj/L⌉ − nj/L) · c^(j/L), which folds 2^n ≡ c into the transform, so a
  plain cyclic convolution of length L gives x² mod (2^n − c) with no zero padding (same idea as Prime95).
* Every transform is done twice in one pass: in plain FP32 (approximate) and exactly in GF(p²), p = 2^31 − 1 (a
  complex-like NTT, i² = −1). The exact convolution value is z = r + p·round((z_fp32 − r)/p) with r = z mod p from
  the NTT, so the FP32 part only has to be within ±p/2 and words hold up to ~17.5 bits.
* GF(p) facts used: 2^31 = 1, so powers of two are 31-bit rotations (1/2 = 2^30, 1/√2 = 2^15); 2 and 25 have odd
  order, so the DWT weights 2^(k/L) and 25^(j/L) exist. Lengths are powers of two (25 has no 3rd or 7th roots and
  GF(p²) has no 5th roots of unity).
* Four-step FFT, W = 512 lines × H columns, three kernels per squaring (`fwdW`, `tail`, `invW`), two-level lazy
  carries, Gerbicz error check every B² iterations (B = 1000), checkpoint file `dfprp_<n>_<c>.ckpt`. Round-off
  above 0.42 inside a Gerbicz block makes that block be redone.
* Digits are stored as 16-bit pairs while every digit fits (n ≤ 16 L), else 32-bit. The carry out of an element
  (about bits/word + log2(c) + 13 bits) is split into a part for the next element and a part for the one after,
  packed into one 64-bit word per element (8 bytes, whatever the word size).
* `hyprp.cl` is written so it compiles as OpenCL C and as CUDA C++ (NVRTC); `gpu_backend.h` has the two backends.
* **FP64 mode** (`--fp64`): the same transform as a plain double-precision complex FFT, no M31 part. It is meant for
  GPUs with full-rate FP64 (P100, V100, A100, H100, B200), which use it by default; everywhere else FP32 + M31 is
  the default (`--fp32` forces it). Both modes give the same residues. With c = 25 the weights c^(j/L) span a factor
  25, which costs FP64 about 2 bits/word, so it holds 0.7–1 bit/word less than FP32 + M31 at the same length.

`dfprp.exe` loads `hyprp.cl` from its own folder at run time; the project copies it to `bin\<Configuration>\`.

## FFT lengths

The length is the smallest power of two whose bits/word stay under the limit below. The limits come from round-off
surveys on a Kaggle T4 and a Radeon 780M (two or three n per point) and give a worst maxerr of about 0.27 at each
cutoff. n within 32 of a multiple of L/4 (more round-off) takes the next length.

| FFT length | bits/word FP32 + M31 | largest n | bits/word FP64 | largest n |
|---|---|---|---|---|
| ≤ 16K | 17.7 | 289,996 (16K) | 17.5 | 286,720 (16K) |
| 32K | 17.7 | 579,993 | 17.25 | 565,248 |
| 64K | 17.7 | 1,159,987 | 17.0 | 1,114,112 |
| 128K | 17.7 | 2,319,974 | 16.75 | 2,195,456 |
| 256K | 17.45 | 4,574,412 | 16.4 | 4,299,161 |
| 512K | 17.2 | 9,017,753 | 16.15 | 8,467,251 |
| 1024K | 16.95 | 17,773,363 | 15.8 | 16,567,500 |
| 2048K | 16.5 | 34,603,008 | 15.5 | 32,505,856 |

The FP64 limits come from round-off surveys on a Radeon 780M (OpenCL). Larger n (longer lengths were not measured)
are refused.

## Running

```
dfprp n [c=25] [-L log2L] [-iters k] [-report k] [-prof] [-B gerbicz_block] [-nogec] [-fresh] [-dev i] [--fp64 | --fp32]
dfprp -bench [-dev i] [-quick] [-out file.json] [--fp64 | --fp32]
```

* `--fp64` / `--fp32` choose the arithmetic; without them it follows the GPU (FP64 on full-rate FP64 GPUs). A
  checkpoint written in one mode resumes in the other when its FFT length is long enough there.

* Every line dfprp prints starts with the UTC time and a level (INFO, WARN, FATAL), as in Mass Prime Siever's log:
  `2026-11-25 06:18:45 UTC+0 [INFO] RES64 7490970688C5F822`.
* No `-iters`: full PRP test with Gerbicz check, resumable from the checkpoint (`-fresh` ignores it).
* `-iters k -report k -prof`: time k squarings and print per-kernel times; `-L k` forces the FFT length 2^k.
* `check.py n [c] [iters]` prints the expected RES64 with Python (small n only).
* Debug's command line is a self-test: `20011 -nogec` must print RES64 `D6077DAFC085AB4C`.
* `HYPRP_BACKEND=opencl` makes a CUDA build use OpenCL on NVIDIA too.

## Benchmark and T4-Hours

The unit of work is the **T4-Hour: what one Tesla T4 card, running alone, computes in one hour.** `dfprp -bench`
measures any GPU for every FFT length from 16K to 2048K, in both storage modes a length uses (16-bit digits for
n ≤ 16 L, 32-bit digits above), at an n inside that mode's range; two rounds, the faster steady-state ms/iter. It
prints the table, the speed relative to the T4 reference, the T4-Hours a test costs (n × T4 ms/iter of its FFT
length and storage mode / 3,600,000), and writes a JSON file. Lengths below 16K are not measured: those tests take
seconds, and their time per iteration is close to the 16K time.

## Kaggle

`kaggle/hyprp_kaggle.ipynb` clones this repository and runs `kaggle/kaggle_run.py`, which builds dfprp with g++
(CUDA + NVRTC when the toolkit is there, OpenCL otherwise). Use a GPU T4 x2 accelerator with Internet on and pick
`MODE = 'selftest'` (build; known residues, including n above 16 bits/word and at 512K; speed of each GPU),
`'verify'` (full tests of exponents Prime95 already did), `'bench'` (the T4-Hour benchmark, one GPU at a time) or
`'run'` (an exponent list; results in `results.json.txt`, checkpoints resume across sessions, one queue per GPU).
The notebook runs whatever is on GitHub, so push changes before running it.

## Speed (one Kaggle T4)

| n | FFT | ms / iteration | per exponent |
|---|---|---|---|
| 3,670,849 / 3,677,545 (full tests, matched Prime95) | 256K | 0.126–0.133 | 7.8–8.1 min |
| 4,000,037 (default settings, both T4s busy) | 256K | 0.112–0.114 | ≈ 7.5 min |
