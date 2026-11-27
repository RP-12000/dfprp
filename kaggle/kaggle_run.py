# Kaggle runner for dfprp (GPU Fermat PRP of 2^n - c, FP32 + M31 hybrid FFT).
# Builds dfprp (sources hyprp.cpp + hyprp.cl) on the Kaggle GPU image (with the CUDA backend when the CUDA toolkit is there, OpenCL otherwise) and
# runs a self-test, a verification against Prime95, the T4-Hour benchmark, or a list of exponents (one queue per
# GPU), resuming checkpoints and results from earlier notebook versions under /kaggle/input.
# Called by kaggle/hyprp_kaggle.ipynb:
#   python kaggle/kaggle_run.py --mode selftest|verify|bench|run [--c 25] [--exponents 3600029,3600061] [--hours 11.5]
import argparse, glob, json, os, re, shutil, subprocess, threading, time

WORK = '/kaggle/working'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = '/kaggle/temp/dfprp-build'
PROG = 'dfprp'   # the program's name: executable, results' "program", checkpoint and log files
SRC = 'hyprp'    # its source files: hyprp.cpp, hyprp.cl
EXE = os.path.join(BUILD, PROG)

KNOWN = {  # Prime95 30.19 PRP-3 residue-type 1 results for 2^n - 25 (from the project's results.json.txt)
    3678495: 'ABC15849A0B8657C',
    3677545: 'CC900B8F8671EBE1',
    3678855: 'C2A776E50ADE26FA',
    3670849: '6809CE838AE81EF9',
}
# RES64 after 3000 iterations, computed with an independent double-float FFT program: checks dfprp above
# 16 bits/word (32-bit digits) and at 512K words
CROSS = {4200013: '7490970688C5F822', 4500013: '386AEF1A83E0FA7C', 4574401: '15011123C7996836',
         9000013: '6B4A2A630EE28A6F'}
REPORT = 10000        # dfprp prints a progress line every REPORT iterations (the Gerbicz check is every 1,000,000)
RUNNING = []          # live dfprp processes, killed on Ctrl+C
LIVE = {}             # GPU -> (n, log path) of the test it is running, for the progress lines
SEEN = {}             # log path -> bytes of it already shown as progress lines
PRECISION = []        # ['--fp64'] or ['--fp32'] to force dfprp's arithmetic (default: dfprp decides)
STOP = threading.Event()
lock = threading.Lock()


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if check and r.returncode:
        print(r.stdout, r.stderr)
        raise SystemExit(f'command failed: {cmd}')
    return r.stdout.strip()


def cuda_flags():
    """compiler/linker flags for dfprp's CUDA backend (driver API + NVRTC), or None"""
    inc = sh("for d in /usr/local/cuda/include /usr/local/cuda-*/include /usr/include; do "
             "[ -f $d/cuda.h ] && [ -f $d/nvrtc.h ] && echo $d && break; done", False)
    nvrtc = sh("find /usr/local/cuda/ /usr/local/cuda-*/ /usr/lib/x86_64-linux-gnu -name 'libnvrtc.so*' "
               "2>/dev/null | grep -v builtins | sort | head -1", False)
    # the driver's own libcuda (Kaggle mounts it with the other NVIDIA driver libraries), else the toolkit's stub
    cuda = sh("for f in /usr/local/nvidia/lib64/libcuda.so.1 /usr/lib64-nvidia/libcuda.so.1 "
              "/usr/lib/x86_64-linux-gnu/libcuda.so.1; do [ -f $f ] && echo $f && break; done", False) or \
        sh("ldconfig -p | grep -o '/[^ ]*libcuda.so.1' | head -1", False) or \
        sh("find /usr/local/cuda/ /usr/local/cuda-*/ -path '*stubs/libcuda.so' 2>/dev/null | head -1", False)
    print('CUDA headers   :', inc or 'NOT FOUND')
    print('libnvrtc       :', nvrtc or 'NOT FOUND')
    print('libcuda        :', cuda or 'NOT FOUND')
    if not (inc and nvrtc and cuda):
        return None
    rp = os.path.dirname(nvrtc) + ('' if 'stubs' in cuda else ':' + os.path.dirname(cuda))
    return f'-DHYPRP_CUDA -I{inc} {nvrtc} {cuda} -Wl,-rpath,{rp}'


def build():
    if subprocess.run('nvidia-smi -L', shell=True, capture_output=True).returncode != 0:
        raise SystemExit('no NVIDIA GPU in this session: Settings -> Accelerator -> GPU T4 x2, then run again')
    os.makedirs(BUILD, exist_ok=True)
    hdr = sh("find /usr/local/cuda/ /usr/include /usr/local/include -path '*CL/cl.h' 2>/dev/null | head -1", False)
    lib = sh("find /usr/local/cuda/ /usr/lib/x86_64-linux-gnu -name 'libOpenCL.so*' 2>/dev/null | sort | head -1", False)
    nvcl = sh("ldconfig -p | grep -o '/[^ ]*libnvidia-opencl.so.1' | head -1", False) or \
        sh("find / -name 'libnvidia-opencl.so.1' 2>/dev/null | head -1", False)
    print('CL/cl.h        :', hdr or 'NOT FOUND')
    print('libOpenCL      :', lib or 'NOT FOUND')
    print('NVIDIA OpenCL  :', nvcl or 'NOT FOUND')
    if not hdr or not lib:
        sh('apt-get -qq update && apt-get -qq install -y ocl-icd-opencl-dev opencl-headers > /dev/null')
        hdr = sh("find /usr/include -path '*CL/cl.h' | head -1")
        lib = sh("find /usr/lib/x86_64-linux-gnu -name 'libOpenCL.so*' | sort | head -1")
    if not nvcl:
        raise SystemExit('the NVIDIA OpenCL driver library is not visible in this container; OpenCL cannot run here')
    icd = '/etc/OpenCL/vendors/nvidia.icd'
    if not os.path.exists(icd):
        os.makedirs(os.path.dirname(icd), exist_ok=True)
        open(icd, 'w').write(nvcl + '\n')
        print('wrote', icd)
    inc = os.path.dirname(os.path.dirname(hdr))
    shutil.copy(os.path.join(REPO, SRC + '.cl'), BUILD)  # the program loads hyprp.cl from the exe's folder
    cmd = (f'g++ -O2 -std=c++17 -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS '
           f'-I{inc} {REPO}/{SRC}.cpp -o {EXE} {lib}')
    cf = cuda_flags()
    built = False
    if cf:
        r = subprocess.run(cmd + ' ' + cf, shell=True, text=True, capture_output=True)
        if r.returncode == 0:
            built = True
            print('built', EXE, '(CUDA on NVIDIA, OpenCL otherwise)')
        else:
            print('CUDA build of dfprp failed, building it with OpenCL only:\n' + (r.stdout + r.stderr)[-3000:])
    if not built:
        sh(cmd)
        print('built', EXE, '(OpenCL)')
    print(sh('nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv', False))


def gpu_count():
    try:
        return max(1, len(sh('nvidia-smi -L').splitlines()))
    except SystemExit:
        return 1


def stamp(level='INFO', dev=None, when=None):
    """'YYYY-MM-DD HH:MM:SS UTC+0 [LEVEL] ', the line prefix of dfprp's and Mass Prime Siever's logs; a GPU's lines
    start '[GPU d YYYY-MM-DD HH:MM:SS] ' instead (UTC); a GPU's warning starts '[WARN YYYY-MM-DD HH:MM:SS] GPU d: '"""
    when = when or time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    if dev is None:
        return f'{when} UTC+0 [{level}] '
    return f'[GPU {dev} {when}] ' if level == 'INFO' else f'[{level} {when}] GPU {dev}: '


def hms(seconds):
    """hh:mm:ss, with 'Nd ' in front from 24 hours on"""
    s = max(0, int(round(seconds)))
    d, s = divmod(s, 86400)
    return (f'{d}d ' if d else '') + f'{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}'


def say(text, level='INFO', dev=None):
    print(stamp(level, dev) + text, flush=True)


def msg(line):
    """a dfprp output line without its 'YYYY-MM-DD HH:MM:SS UTC+0 [LEVEL] ' time stamp"""
    m = re.match(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC\+0 \[\w+\] ', line)
    return line[m.end():] if m else line


def parse(log):
    res64 = re.search(r'RES64 ([0-9A-F]{16})', log)
    res2048 = re.search(r'RES2048 ([0-9A-F]+)', log)
    verdict = re.search(r'is (a probable prime!|composite)', log)
    ms = re.search(r'\(([0-9.]+) ms/iter', log)
    return (res64.group(1) if res64 else None, res2048.group(1) if res2048 else None,
            verdict.group(1) if verdict else None, float(ms.group(1)) if ms else None)


def short(dev, n, line):
    """a dfprp log line as a short progress note (None for the lines not shown), e.g.
    '[GPU 0 2026-11-27 02:20:42] Iteration 2500000/4109583 [60.83%], ms/iter: 0.1313, ETA: 00:03:31'"""
    m = msg(line)
    lv = re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC\+0 \[(\w+)\] ', line)
    t = stamp(lv.group(2), dev, lv.group(1)) if lv else stamp('INFO', dev)
    p = re.search(r'iter (\d+)/(\d+) +([0-9.]+) ms/iter', m)
    if p:
        a, todo, ms = int(p.group(1)), int(p.group(2)), float(p.group(3))
        return (f'{t}Iteration {a}/{n} [{100 * a / n:.2f}%], ms/iter: {ms:.4f}, '
                f'ETA: {hms(ms * (todo - a) / 1000)}')
    if m.startswith(('Gerbicz', 'resumed', 'round-off', 'using the checkpoint')):
        return f'{t}n={n} {m}'
    return None


def show_progress():
    """prints the new progress lines of every running test: dfprp writes one every REPORT iterations and runs a
    Gerbicz error check every 1,000,000. All new lines go out in one write, so Kaggle's log has no blank lines
    between them."""
    with lock:
        live = sorted(LIVE.items())
    out = []
    for dev, (n, path) in live:
        try:
            with open(path, 'rb') as f:
                f.seek(SEEN.get(path, 0))
                data = f.read()
        except OSError:
            continue
        cut = data.rfind(b'\n') + 1  # whole lines only; the rest is read next time
        SEEN[path] = SEEN.get(path, 0) + cut
        for line in data[:cut].decode('utf-8', 'replace').splitlines():
            s = short(dev, n, line)
            if s:
                out.append(s)
    if out:
        print('\n'.join(out), flush=True)


def run_one(dev, n, c, args, cwd, logname, deadline):
    os.makedirs(cwd, exist_ok=True)
    cmd = [EXE, str(n), str(c), '-dev', str(dev)] + args + PRECISION
    t0 = time.time()
    path = os.path.join(cwd, logname)
    with open(path, 'w') as lf:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        RUNNING.append(p)
        with lock:
            LIVE[dev] = (n, path)
            SEEN[path] = 0
        try:
            while p.poll() is None:
                if STOP.is_set() or (deadline and time.time() > deadline):
                    p.kill()
                    p.wait()
                    return None, time.time() - t0
                time.sleep(2)
        finally:
            RUNNING.remove(p)
            with lock:
                LIVE.pop(dev, None)
    return open(path).read(), time.time() - t0


def tail(log, k=12):
    return '\n'.join('    ' + l for l in (log or '(no output: timed out)').strip().splitlines()[-k:])


def selftest(c):
    """known residues on GPU 0, then the speed of every GPU (the backend dfprp uses here: CUDA on NVIDIA)"""
    # full test with Gerbicz checks (small n)
    log, _ = run_one(0, 20011, 25, ['-fresh', '-B', '30'], f'{WORK}/selftest', 'n20011.log', time.time() + 120)
    be = re.search(r'\| (CUDA|OpenCL)', log or '')
    print('backend:', be.group(1) if be else '?')
    r = parse(log or '')[0]
    print(f'n=20011 RES64 {r}', 'OK' if r == 'D6077DAFC085AB4C' else 'WRONG (expected D6077DAFC085AB4C)')
    if r != 'D6077DAFC085AB4C':
        print(tail(log))
    # above 16 bits/word (32-bit digits) and at 512K, against an independent program's results
    for n, want in CROSS.items():
        log, _ = run_one(0, n, 25, ['-iters', '3000', '-report', '3000'], f'{WORK}/selftest', f'x{n}.log',
                         time.time() + 120)
        r = parse(log or '')[0]
        opt = re.search(r'options: ([^\n]+)', log or '')
        print(f'n={n} 3000 iterations RES64 {r}', 'OK' if r == want else f'WRONG (expected {want})',
              f'[{opt.group(1)}]' if opt else '')
    for dev in range(gpu_count()):
        log, _ = run_one(dev, 4000037, c, ['-iters', '20000', '-report', '20000', '-prof'], f'{WORK}/selftest',
                         f'speed_gpu{dev}.log', time.time() + 120)
        print(f'--- GPU {dev}, n=4000037, 20000 iterations')
        for line in (log or '').splitlines():
            if msg(line).startswith(('n=', 'device', 'options', 'plain', 'kernel', 'RES64')):
                print(line)
        ms = parse(log or '')[3]
        if ms:
            print(f'GPU {dev}: {ms:.4f} ms/iter -> about {ms * 4_000_000 / 60000:.1f} min per 4M-bit exponent')
        else:
            print(tail(log))


def bench_all():
    """dfprp -bench on every GPU, one GPU at a time (1 T4-Hour = what one T4 card computes in one hour);
    JSON results in /kaggle/working"""
    os.makedirs(WORK, exist_ok=True)
    for dev in range(gpu_count()):
        cmd = [EXE, '-bench', '-dev', str(dev), '-out', f'{WORK}/{PROG}_bench_gpu{dev}.json'] + PRECISION
        r = subprocess.run(cmd, cwd=WORK, text=True, capture_output=True, timeout=1800)
        print(f'===== GPU {dev}')
        print('\n'.join(l for l in (r.stdout + r.stderr).splitlines() if not msg(l).startswith('round ')), flush=True)


def restore():
    """copy checkpoints and earlier results from /kaggle/input into /kaggle/working"""
    done = {}
    for f in glob.glob('/kaggle/input/**/results.json.txt', recursive=True):
        for line in open(f):
            try:
                d = json.loads(line)
                done[(d['n'], d['c'])] = d
            except Exception:
                pass
    os.makedirs(WORK, exist_ok=True)
    with open(f'{WORK}/results.json.txt', 'w') as out:
        for d in done.values():
            out.write(json.dumps(d) + '\n')
    # checkpoints saved before the rename are called hyprp_<n>_<c>.ckpt (same contents); a dfprp_ one comes first
    for f in sorted(glob.glob('/kaggle/input/**/*_*.ckpt', recursive=True),
                    key=lambda f: not os.path.basename(f).startswith(PROG + '_')):
        name = os.path.basename(f)
        if not name.startswith((PROG + '_', 'hyprp_')):
            continue
        dst = os.path.join(WORK, 'ckpt', PROG + name[name.index('_'):])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy(f, dst)
    say(f'restored {len(done)} results and {len(glob.glob(WORK + "/ckpt/*.ckpt"))} checkpoints')
    return done


def worker(dev, queue, c, deadline, done):
    ck = os.path.join(WORK, 'ckpt')  # all GPUs share one checkpoint folder; each exponent runs on one GPU
    while True:
        with lock:
            if not queue:
                return
            n = queue.pop(0)
        if (n, -c) in done:
            continue
        log, secs = run_one(dev, n, c, ['-report', str(REPORT)], ck, f'{PROG}_{n}.log', deadline)
        if log is None:
            why = 'stopped' if STOP.is_set() else 'time limit'
            say(f'{why}: n={n} interrupted, its checkpoint (saved every 1M iterations) is kept', dev=dev)
            return
        res64, res2048, verdict, ms = parse(log)
        if not res64 or not verdict:
            say(f'n={n} failed:\n' + log[-2000:], 'WARN', dev)
            continue
        be = re.search(r'\| (CUDA|OpenCL)', log)
        uu = re.search(r'\| uuid (GPU-[0-9a-f-]+)', log)  # the physical card: the "computer" of a Kaggle/Colab result
        cs = re.search(r'checksum ([0-9A-F]{8})', log)    # dfprp's CRC-32 of the result (guards against slips)
        ver = re.search(r'Starting dfprp v([0-9.]+)', log)
        fl = re.search(r' L=(\d+) ', log)                   # FFT length in words, as Prime95's "fft-length"
        d = {'n': n, 'k': 1, 'b': 2, 'c': -c, 'worktype': 'PRP-3', 'residue-type': 1,
             'status': 'P' if verdict.startswith('a probable') else 'C', 'res64': res64, 'res2048': res2048,
             'program': PROG, 'backend': be.group(1) if be else None, 'gpu': dev,
             'gpu_uuid': uu.group(1) if uu else None, 'version': ver.group(1) if ver else None,
             'checksum': cs.group(1) if cs else None, 'fft-length': int(fl.group(1)) if fl else None,
             'proof': {'version': None, 'power': None, 'hashsize': None, 'md5': None},  # Prime95's keys; no proof file yet
             'seconds': round(secs, 1),
             'ms_per_iter': ms, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}
        known = KNOWN.get(n) if c == 25 else None
        if known:
            d['prime95_res64'] = known
            d['matches_prime95'] = known == res64
        with lock:
            done[(n, -c)] = d
            with open(f'{WORK}/results.json.txt', 'a') as out:
                out.write(json.dumps(d) + '\n')
        flag = '' if not known else ('  MATCHES Prime95' if known == res64 else f'  DIFFERS from Prime95 {known}')
        say(f'n={n} {"PRP!" if d["status"] == "P" else "composite"} RES64 {res64} '
            f'{secs / 60:.1f} min{flag}', dev=dev)


def main(mode, exponents, c, hours):
    t_start = time.time()
    # processes left over from an interrupted run would fight over the GPU and the checkpoint files
    if subprocess.run(['pkill', '-f', BUILD + '/'], capture_output=True).returncode == 0:
        say('stopped dfprp processes left over from an earlier run (they resume from their checkpoints)')
        time.sleep(2)
    build()
    if mode == 'selftest':
        selftest(c)
        return
    if mode == 'bench':
        bench_all()
        print(f'benchmark finished in {time.time() - t_start:.0f} s')
        return
    if mode == 'verify':
        exponents = sorted(KNOWN)[:gpu_count()] if c == 25 else exponents
    done = restore()
    queue = [n for n in exponents if (n, -c) not in done]
    ngpu = gpu_count()
    deadline = t_start + hours * 3600
    say(f'{len(queue)} exponents to test on {ngpu} GPU(s); stopping at {hours} h')
    say(f'progress every {REPORT} iterations, Gerbicz error check every 1,000,000; '
        f'full logs in {WORK}/ckpt/{PROG}_<n>.log')
    ts = [threading.Thread(target=worker, args=(dev, queue, c, deadline, done), daemon=True)
          for dev in range(ngpu)]
    for t in ts:
        t.start()
    try:
        while any(t.is_alive() for t in ts):
            time.sleep(1)
            show_progress()
    except KeyboardInterrupt:
        STOP.set()
        for p in list(RUNNING):
            p.kill()
        for t in ts:
            t.join(10)
        say('interrupted: tests stopped; run again to resume from the checkpoints')
        return
    say(f'finished: {len(done)} results in {WORK}/results.json.txt, {len(queue)} exponents not started')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['selftest', 'verify', 'bench', 'run'], default='selftest')
    ap.add_argument('--c', type=int, default=25)
    ap.add_argument('--exponents', default='', help='comma or space separated list of n')
    ap.add_argument('--hours', type=float, default=11.5)
    ap.add_argument('--precision', choices=['auto', 'fp64', 'fp32'], default='auto',
                    help='dfprp arithmetic: auto = FP64 on full-rate FP64 GPUs (A100, H100, ...), FP32+M31 elsewhere')
    a = ap.parse_args()
    exps = [int(x) for x in re.split(r'[,\s]+', a.exponents) if x.strip().isdigit()]
    for f in glob.glob('/kaggle/input/**/exponents.txt', recursive=True):
        exps += [int(x) for x in open(f).read().split() if x.strip().isdigit()]
    if a.precision != 'auto':
        PRECISION.append('--' + a.precision)
    main(a.mode, list(dict.fromkeys(exps)), a.c, a.hours)
