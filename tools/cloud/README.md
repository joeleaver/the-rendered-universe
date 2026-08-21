# Running the heavy parts on rented CPU

Everything in this repository through part 34 runs on a laptop with
numpy and Pillow. Two measurements do not: the L = 6 exact-rank
validation anchor of part 34 and the L = 8/10/12 walk beyond the
exact-diagonalization wall. This directory holds the machinery for
running those on rented hardware, plus the failure modes we paid
for — recorded so the next attempt pays for none of them again.

Scripts here were used against JarvisLabs CPU VMs (`jl` CLI, 32
vCPU / 128 GB, ~$0.79/h) but nothing is vendor-specific beyond the
`jl` calls in the watchdog.

## The runbook

```bash
jl create --vm --cpu --vcpus 32 --ram 128 --region IN2 --yes --json
#   -> machine_id, ssh_command; IN1 may refuse 32 contiguous cores

# 1. bundle: massgap.py, dmrg.py, observatory/{__init__,mps}.py
scp -r cloudrun ubuntu@<ip>:/home/ubuntu/
scp build_b2.sh run_stage.sh ubuntu@<ip>:/home/ubuntu/

# 2. toolchain + micromamba (bzip2 is NOT preinstalled)
ssh ubuntu@<ip> 'sudo apt-get update -q && \
    sudo apt-get install -y -q build-essential cmake bzip2 && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj bin/micromamba'

# 3. build block2 from source (~20 min at -j30), ends with a real
#    L = 4 compute test and a BUILD_OK sentinel
ssh ubuntu@<ip> 'nohup bash build_b2.sh > b2_build.log 2>&1 &'

# 4. stages, ONE PER PROCESS (see lesson 6)
ssh ubuntu@<ip> 'nohup bash run_stage.sh anchors  > gate.log 2>&1 &'
ssh ubuntu@<ip> 'nohup bash run_stage.sh walk 8 1000 > w8.log 2>&1 &'

# 5. when done
jl destroy <machine_id> --yes --json
```

Install `jl_watchdog.sh` locally *before* starting anything:

```bash
cp jl_watchdog.sh ~/.local/bin/ && chmod +x ~/.local/bin/jl_watchdog.sh
mkdir -p ~/.jl_watchdog && touch ~/.jl_watchdog/heartbeat
(crontab -l 2>/dev/null; echo "*/20 * * * * \$HOME/.local/bin/jl_watchdog.sh") | crontab -
```

It pauses every Running instance when the managing session's
heartbeat goes stale (3 h) **or** when a box has had no compute
process for 40 minutes. Pause preserves disk; destroy is never
automated. The heartbeat is `~/.jl_watchdog/heartbeat` and only a
live session may touch it — anything automated touching it defeats
the switch.

## Lessons, in the order they cost us

1. **The pip `block2` wheel is broken on this stack.** Its vendored
   `libmkl_core` and its pinned `mkl` kernel package are different
   versions: `libmkl_def.so.1: undefined symbol:
   mkl_sparse_optimize_bsr_trsm_i8`. `LD_LIBRARY_PATH` does not fix
   it (MKL dlopens kernels by absolute path inside `block2.libs/`),
   and neither does symlinking them in. conda-forge has no block2
   package at all. **Build from source.**
2. **Source build needs three things the docs don't say:** MKL from
   pip *inside the same env* (so core and kernels cannot disagree),
   `CMAKE_POLICY_VERSION_MINIMUM=3.5` for modern cmake, and — on
   GCC 13 — `-Wno-error=stringop-overflow -Wno-error=array-bounds
   -Wno-error=maybe-uninitialized`, because block2 compiles with
   `-Werror` and GCC 13 emits a false positive on an inlined
   `memmove`. (GCC 15 does not; that is why the local build worked.)
3. **Silence is not success.** Two separate runs died with empty or
   truncated logs: a `jl run -- sh -lc '...'` whose nested quoting
   swallowed the command, and a build whose `apt-get ... > /dev/null
   2>&1 || true` hid that `cmake` was never installed. **Ship a
   script and run `bash script.sh`; never a quoted command line.**
   Put an explicit sentinel (`BUILD_OK` / `BUILD_FAILED`) at the end
   of every stage and monitor for *both*.
4. **Python buffers, and buffered output dies with the process.**
   Always `python -u`. An 18-hour run reported nothing at all
   because its prints never flushed; the numbers had to be recovered
   from the saved MPS files afterward.
5. **block2's scratch directory is process-global.** Two block2
   processes sharing one scratch corrupt each other's tensor files
   (`DataFrame::load_data ... failed`, then an MKL error storm). A
   concurrent smoke test killed a 26-hour run this way. `b2_run`
   now uses `/tmp/b2_smg_<pid>`.
6. **block2 pre-allocates a fixed memory pool per driver.** The 1 GB
   default aborts a χ = 4200 run mid-sweep (`exceeding allowed
   memory`), and `B2_STACK_GB` sizes it — but each `b2_run` builds a
   *new* driver, so a five-run stage requests the pool five times
   and the box dies **silently**, no traceback, no OOM line. Run one
   stage per process, and keep the pool well under RAM/5.
7. **Test safety code in the environment it will run in.** The
   watchdog did `command -v jl || exit 0`; cron's PATH is
   `/usr/bin:/bin`, which excludes `~/.local/bin`. It ran every 20
   minutes for 16 hours and silently did nothing while a crashed job
   left a box billing. Verify with `env -i HOME=$HOME
   PATH=/usr/bin:/bin bash jl_watchdog.sh`.
8. **A live box is not a working box.** Heartbeat staleness only
   catches an absent operator; it does not catch a crashed job on a
   running machine — which is what actually wasted the money. Hence
   the idle check.
9. **`pgrep -f <pattern>` matches the shell running it.** The remote
   `sh -c 'pgrep -cf "massgap.py"'` counts itself, so an idle box
   reads as busy forever. Use the bracket trick: `[m]assgap[.]py`.
10. **`jl resume` returns a NEW machine_id.** Always use the
    returned value; the old id is dead, and pausing it silently does
    nothing.

## What is still owed

Task: the L = 6 exact-rank anchor (χ ≈ 4200, closes part 34's
validation gate) and the L = 8/10/12 walk in both gap channels.
Estimated 1–2 days on one 32-vCPU box, ~$20–40. Fix lesson 6 first:
run each anchor and each walk point as its own process invocation
(`--dmrg walk L CHI` already does exactly this) with
`B2_STACK_GB` ≈ 20.
