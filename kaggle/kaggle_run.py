# dfprp Kaggle runner: build dfprp from this repository on the Kaggle GPU image and run self-test /
# verification / PRP tests, one exponent queue per GPU, resuming checkpoints and results from earlier notebook
# versions under /kaggle/input.  Called by kaggle/dfprp_kaggle.ipynb:
#   python kaggle/kaggle_run.py --mode selftest|verify|run [--c 25] [--exponents 3600029,3600061] [--hours 11.5]
import argparse, glob, json, os, re, shutil, subprocess, sys, threading, time

WORK = '/kaggle/working'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = '/kaggle/temp/dfprp-build'
EXE = os.path.join(BUILD, 'dfprp')

KNOWN = {  # Prime95 30.19 PRP-3 residue-type 1 results for 2^n - 25 (from the project's results.json.txt)
    3678495: 'ABC15849A0B8657C',
    3677545: 'CC900B8F8671EBE1',
    3678855: 'C2A776E50ADE26FA',
    3670849: '6809CE838AE81EF9',
}


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if check and r.returncode:
        print(r.stdout, r.stderr)
        raise SystemExit(f'command failed: {cmd}')
    return r.stdout.strip()


def build():
    os.makedirs(BUILD, exist_ok=True)
    shutil.copy(os.path.join(REPO, 'dfprp.cl'), BUILD)  # dfprp loads its kernel from the exe's folder
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
    sh(f'g++ -O2 -std=c++17 -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS '
       f'-I{inc} {REPO}/dfprp.cpp -o {EXE} {lib}')
    print('built', EXE)
    print(sh('nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv', False))


def gpu_count():
    try:
        return max(1, len(sh('nvidia-smi -L').splitlines()))
    except SystemExit:
        return 1


def parse(log):
    res64 = re.search(r'RES64 ([0-9A-F]{16})', log)
    res2048 = re.search(r'RES2048 ([0-9A-F]+)', log)
    verdict = re.search(r'is (a probable prime!|composite)', log)
    ms = re.search(r'\(([0-9.]+) ms/iter', log)
    return (res64.group(1) if res64 else None, res2048.group(1) if res2048 else None,
            verdict.group(1) if verdict else None, float(ms.group(1)) if ms else None)


def run_one(dev, n, c, args, cwd, logname, deadline):
    os.makedirs(cwd, exist_ok=True)
    cmd = [EXE, str(n), str(c), '-dev', str(dev)] + args
    t0 = time.time()
    with open(os.path.join(cwd, logname), 'w') as lf:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        while p.poll() is None:
            if deadline and time.time() > deadline:
                p.kill()
                p.wait()
                return None, time.time() - t0
            time.sleep(2)
    return open(os.path.join(cwd, logname)).read(), time.time() - t0


def selftest(c):
    log, _ = run_one(0, 20011, 25, ['-nogec'], f'{WORK}/selftest', 'n20011.log', None)
    r = parse(log)
    print('n=20011 RES64', r[0], 'OK' if r[0] == 'D6077DAFC085AB4C' else 'WRONG (expected D6077DAFC085AB4C)')
    for dev in range(gpu_count()):
        log, _ = run_one(dev, 4000037, c, ['-iters', '20000', '-report', '20000', '-prof'],
                         f'{WORK}/selftest', f'bench_gpu{dev}.log', None)
        print(f'--- GPU {dev}, n=4000037, 20000 iterations')
        for line in log.splitlines():
            if line.startswith(('n=', 'device', 'plain', 'kernel', 'RES64')):
                print(line)
        ms = parse(log)[3]
        if ms:
            print(f'GPU {dev}: {ms:.4f} ms/iter -> about {ms * 4_000_000 / 60000:.1f} min per 4M-bit exponent')


def restore():
    """copy checkpoints and earlier results from /kaggle/input into /kaggle/working"""
    done = {}
    for f in glob.glob('/kaggle/input/**/results.jsonl', recursive=True):
        for line in open(f):
            try:
                d = json.loads(line)
                done[(d['n'], d['c'])] = d
            except Exception:
                pass
    os.makedirs(WORK, exist_ok=True)
    with open(f'{WORK}/results.jsonl', 'w') as out:
        for d in done.values():
            out.write(json.dumps(d) + '\n')
    for f in glob.glob('/kaggle/input/**/dfprp_*.ckpt', recursive=True):
        dst = os.path.join(WORK, 'ckpt', os.path.basename(f))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy(f, dst)
    print(f'restored {len(done)} results and {len(glob.glob(WORK + "/ckpt/*.ckpt"))} checkpoints')
    return done


lock = threading.Lock()


def worker(dev, queue, c, deadline, done):
    ck = os.path.join(WORK, 'ckpt')  # all GPUs share one checkpoint folder; each exponent runs on one GPU
    while True:
        with lock:
            if not queue:
                return
            n = queue.pop(0)
        if (n, -c) in done:
            continue
        print(f'[GPU {dev}] start n={n}', flush=True)
        log, secs = run_one(dev, n, c, [], ck, f'dfprp_{n}.log', deadline)
        if log is None:
            print(f'[GPU {dev}] time limit: n={n} stopped, checkpoint kept for the next session', flush=True)
            return
        res64, res2048, verdict, ms = parse(log)
        if not res64 or not verdict:
            print(f'[GPU {dev}] n={n} failed:\n' + log[-2000:], flush=True)
            continue
        d = {'n': n, 'k': 1, 'b': 2, 'c': -c, 'worktype': 'PRP-3', 'residue-type': 1,
             'status': 'P' if verdict.startswith('a probable') else 'C', 'res64': res64, 'res2048': res2048,
             'program': 'dfprp', 'gpu': dev, 'seconds': round(secs, 1), 'ms_per_iter': ms,
             'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}
        known = KNOWN.get(n) if c == 25 else None
        if known:
            d['prime95_res64'] = known
            d['matches_prime95'] = known == res64
        with lock:
            done[(n, -c)] = d
            with open(f'{WORK}/results.jsonl', 'a') as out:
                out.write(json.dumps(d) + '\n')
        flag = '' if not known else ('  MATCHES Prime95' if known == res64 else f'  DIFFERS from Prime95 {known}')
        print(f'[GPU {dev}] n={n} {"PRP!" if d["status"] == "P" else "composite"} RES64 {res64} '
              f'{secs / 60:.1f} min{flag}', flush=True)


def main(mode, exponents, c, hours):
    t_start = time.time()
    build()
    if mode == 'selftest':
        selftest(c)
        return
    if mode == 'verify':
        exponents = sorted(KNOWN)[:gpu_count()] if c == 25 else exponents
    done = restore()
    queue = [n for n in exponents if (n, -c) not in done]
    ngpu = gpu_count()
    deadline = t_start + hours * 3600
    print(f'{len(queue)} exponents to test on {ngpu} GPU(s); stopping at {hours} h')
    ts = [threading.Thread(target=worker, args=(dev, queue, c, deadline, done)) for dev in range(ngpu)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f'finished: {len(done)} results in {WORK}/results.jsonl, {len(queue)} exponents not started')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['selftest', 'verify', 'run'], default='selftest')
    ap.add_argument('--c', type=int, default=25)
    ap.add_argument('--exponents', default='', help='comma or space separated list of n')
    ap.add_argument('--hours', type=float, default=11.5)
    a = ap.parse_args()
    exps = [int(x) for x in re.split(r'[,\s]+', a.exponents) if x.strip().isdigit()]
    for f in glob.glob('/kaggle/input/**/exponents.txt', recursive=True):
        exps += [int(x) for x in open(f).read().split() if x.strip().isdigit()]
    main(a.mode, list(dict.fromkeys(exps)), a.c, a.hours)
