# GpuPRP

GPU Fermat PRP test (base 3, same residue type as Prime95 "PRP-3, residue-type 1") for numbers N = 2^n − c,
written in OpenCL for the Radeon 780M in this laptop. Open `GpuPRP.slnx` in Visual Studio 2026.

## Projects

| Project | Folder | What it is |
|---|---|---|
| `gpuprp` | root | v1 prototype: zero-padded complex FFT, 18 bits/word |
| `prp2` | `v2` | FP64 weighted real FFT (no zero padding), Gerbicz check, checkpoints |
| `prp3` | `v2` | same as prp2 but every float is a double-float (pair of FP32), lengths 2^k, 3·2^k, 5·2^k |
| `prp4` | `v2` | exact integer NTT over three 31-bit primes + CRT, 32-bit limbs |
| `mbench` | `v2` | memory bandwidth / FP64 / LDS micro-benchmark |
| `mbench2` | `v2` | butterfly throughput: FP64 vs double-float vs 32-bit NTT |

Each `prpN.exe` loads its kernel `prpN.cl` from the folder of the exe at run time; the projects copy the `.cl`
next to the exe on build (`bin\Debug` or `bin\Release`). Edit the `.cl` and just re-run: no rebuild needed for
kernel-only changes, but the copy in `bin\...` is what gets loaded, so build (or copy) after editing.

OpenCL headers and `OpenCL.lib` come from vcpkg (`C:\Users\kelvi\vcpkg\installed\x64-windows`, set in
`GpuPRP.props` as `VcpkgInstalled`). The OpenCL runtime is the system `OpenCL.dll` from the AMD driver.

## Running

```
prp2 n [c=25] [-iters k] [-prof] [-B gerbicz_block] [-nogec] [-fresh] [-report k] [-inject it]
prp3 n [c=25] ...   (PRP3_LM=5 with -L 16 selects a 5*2^16 length)
prp4 n [c=25] ...   (best on the 780M: PRP4_W=1024 PRP4_WG=256 PRP2_TG=8)
```

* No `-iters`: full PRP test with Gerbicz check and a checkpoint file `prpN_<n>_<c>.ckpt` in the working dir.
* `-iters k -report k -prof`: benchmark k squarings and print per-kernel times.
* `v2\check.py n [c] [iters]` prints the expected RES64 with Python (small n only).

The Debug configuration's command line is a quick self-test (`20011 -nogec`, expected RES64
`D6077DAFC085AB4C`). Measure speed only with the Release build, and only while Prime95 is stopped.

## Results on the Radeon 780M (n = 4,000,037, Prime95 stopped)

| Version | ms / iteration | time per 4M-bit number |
|---|---|---|
| prp3 double-float, 320K FFT | 0.397 | ~26.5 min |
| prp2 FP64, 256K FFT | 0.452 | ~30 min |
| prp4 integer NTT, 256K | 0.51 | ~34 min |

A full prp2 run of n = 3678495 took 33.2 min and matched Prime95's res64 and res2048.
