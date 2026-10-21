#!/usr/bin/env python3
"""Cycle-level model of one Maxwell SMM executing the steady-state loop of a
NAK-compiled kernel, driven by NAK's own control annotations
(delay=, wt=, rd:, wr:) from the 'NAK IR after calc_instr_deps' dump.

Model (deliberately simple, every knob is a CLI flag so causes can be toggled):
  * 4 schedulers, warps assigned round-robin; one instruction issued per
    scheduler per cycle (no dual issue); loose round-robin among ready warps.
  * After issuing an instruction a warp waits 'delay' cycles (NAK stall count)
    before its next instruction can issue; wt=mask waits for scoreboards.
  * FFMA register-bank conflict (Maxwell: bank = reg % 4, operand reuse cache
    honoured via the .reuse flags from the SASS) costs +1 issue cycle.
  * Shared memory: one SM-wide pipe, 128 B/clk; each LDS/STS costs the number
    of bank-cycles given by the per-kernel cost function (broadcast aware);
    result available LDS_BASE cycles after service.  wr: scoreboard clears at
    completion, rd: scoreboard clears when the pipe starts serving the op.
  * Global loads: LSU pipe at 128 B/clk of address processing (4 cycles for a
    32-lane LDG.128), then an optional L2 line-bandwidth server, then a fixed
    latency chosen per load class (A or B operand) by the cost function.
  * bar.sync: all warps of a workgroup must arrive; membar: just its delay.

Usage: smsim.py <kernel> [--L_dram N] [--L_A N] [--l2_bpc N] [--no-delays]
       [--lds-free] [--mem-free] [--no-conflicts] [--warps-per-wg N] [--wgs N]
       [--iters N] [--ffma-only]
Kernels are described in KERNELS below (file, loop blocks, warp counts, cost fns).
"""
import argparse
import collections
import re
import sys

sys.path.insert(0, "/var/home/marc/.claude/jobs/7dc3895b/tmp/analysis")
import nakdeps  # noqa: E402
import loopstats  # noqa: E402

MEAS = "/var/home/marc/.claude/jobs/7dc3895b/tmp/measured/"
SWEEP = "/var/home/marc/.claude/jobs/7dc3895b/tmp/sweep/"
ADDR = re.compile(r"\[(r\d+)(?:\.\.\d+)?(?:\+(0x[0-9a-f]+))?\]")


def lds_pascal(base, imm):      # A: r13 (broadcast within warp) ; B: r14 (32 distinct float4)
    return 1 if base == "r13" else 4


def lds_rdna3(base, imm):       # A: r42 (2 addrs, same bank) ; B: r41 (16 distinct float4)
    return 2


def lds_r64x64x16d1(base, imm):  # A: r13 (2 addrs same bank) ; B: r14 (16 distinct)
    return 2


def ldg_class_shared(base, imm):   # all tile loads stream from DRAM
    return "dram"


KERNELS = {
    # name: (nak log, sass, loop ranges in sass, loop block ids, warps/WG, WGs/SM, lanes, lds cost fn, ldg class fn)
    "pascal": dict(log=MEAS + "pascal.nak.log", sass=MEAS + "pascal.sass", ranges="268-1e20,38c8-3a20",
                   blocks=[1, 4], warps_per_wg=8, wgs=1, lanes=32, lds=lds_pascal,
                   ldgclass=lambda dst, srcs: "dram"),
    "rdna3": dict(log=MEAS + "rdna3.nak.log", sass=MEAS + "rdna3.sass", ranges="4a8-1fd0,51a8-51d0",
                  blocks=[1, 4], warps_per_wg=4, wgs=2, lanes=32, lds=lds_rdna3,
                  ldgclass=lambda dst, srcs: "dram"),
    # valhall: wg 8x2 = 16 lanes.  A loads: [r26..]/[r18..] via c[0x1][0x0] (A base) ; B via c[0x1][0x10]
    "valhall": dict(log=MEAS + "valhall.nak.log", sass=MEAS + "valhall.sass", ranges="108-a18,bc8-bf8",
                    blocks=[1, 4], warps_per_wg=1, wgs=32, lanes=16, lds=None, ldgclass="by_cbuf", ty_per_warp=2),
    "adreno": dict(log=MEAS + "adreno.nak.log", sass=MEAS + "adreno.sass", ranges="308-1180",
                   blocks=[1], warps_per_wg=2, wgs=6, lanes=32, lds=None, ldgclass="by_cbuf"),
    "r64x64x8d1": dict(log=SWEEP + "r64x64x8d1.nak.log", sass=SWEEP + "r64x64x8d1.sass",
                       ranges="288-1178,1fe8-20a0", blocks=[1, 4], warps_per_wg=4, wgs=5, lanes=32,
                       lds=lambda b, i: 2, ldgclass=lambda dst, srcs: "dram"),
    "r64x64x16d2": dict(log=SWEEP + "r64x64x16d2.nak.log", sass=SWEEP + "r64x64x16d2.sass",
                        ranges="388-22b8,5688-5838", blocks=[1, 4], warps_per_wg=4, wgs=3, lanes=32,
                        lds=lambda b, i: 2, ldgclass=lambda dst, srcs: "dram"),
    "r64x64x32d1": dict(log=SWEEP + "r64x64x32d1.nak.log", sass=SWEEP + "r64x64x32d1.sass",
                        ranges="388-3738,68e8-6ab8", blocks=[1, 4], warps_per_wg=4, wgs=2, lanes=32,
                        lds=lambda b, i: 2, ldgclass=lambda dst, srcs: "dram"),
    "r32x64x16d1": dict(log=SWEEP + "r32x64x16d1.nak.log", sass=SWEEP + "r32x64x16d1.sass",
                        ranges="208-10d8,1f08-1fd8", blocks=[1, 4], warps_per_wg=4, wgs=4, lanes=32,
                        lds=lambda b, i: 1 if b == "r13" else 2, ldgclass=lambda dst, srcs: "dram"),
    "r128x128x8d1": dict(log=SWEEP + "r128x128x8d1.nak.log", sass=SWEEP + "r128x128x8d1.sass",
                         ranges="3e8-1ca8,36e8-3778", blocks=[1, 4], warps_per_wg=8, wgs=1, lanes=32,
                         lds=lambda b, i: 2, ldgclass=lambda dst, srcs: "dram"),
    "r64x64x16d1": dict(log=SWEEP + "r64x64x16d1.nak.log", sass=SWEEP + "r64x64x16d1.sass",
                        ranges="2e8-1e68,3908-3a30", blocks=[1, 4], warps_per_wg=4, wgs=4, lanes=32,
                        lds=lds_r64x64x16d1, ldgclass=lambda dst, srcs: "dram"),
}


def load_ir(path, block_ids):
    """Return the loop instruction list (blocks concatenated) with parsed fields."""
    text = open(path).read()
    idx = text.rfind(nakdeps.SECTION)
    lines = text[idx + len(nakdeps.SECTION):].splitlines()
    blocks = {}
    cur = None
    for line in lines:
        m = nakdeps.BLOCK.match(line)
        if m:
            cur = int(m.group(2))
            blocks[cur] = []
            continue
        if nakdeps.END.match(line):
            cur = None
            continue
        if cur is None:
            if line.startswith("Num GPRs"):
                break
            continue
        m = nakdeps.INS.match(line)
        if not m:
            continue
        dst, op, srcs, ann = m.group(1) or "", m.group(2), m.group(3), m.group(4)
        d = nakdeps.DELAY.search(ann)
        wt = nakdeps.WT.search(ann)
        rd = nakdeps.RD.search(ann)
        wr = nakdeps.WR.search(ann)
        blocks[cur].append(dict(
            dst=dst, op=op, srcs=srcs,
            delay=int(d.group(1)) if d else 1,
            wt=[5 - i for i, c in enumerate(wt.group(1)) if c == "1"] if wt else [],
            rd=int(rd.group(1)) if rd else None,
            wr=int(wr.group(1)) if wr else None,
        ))
    out = []
    for b in block_ids:
        out.extend(blocks[b])
    return out


def classify(i):
    op = i["op"]
    if op.startswith("ffma") or op.startswith("fmul") or op.startswith("fadd"):
        return "ffma" if op.startswith("ffma") else "alu"
    if op.startswith("ld.shared"):
        return "lds"
    if op.startswith("st.shared"):
        return "sts"
    if op.startswith("ld.global"):
        return "ldg"
    if op.startswith("st.global"):
        return "stg"
    if op.startswith("bar"):
        return "bar"
    if op.startswith("membar"):
        return "membar"
    if op == "nop":
        return "nop"
    return "alu"


def ldg_class_by_cbuf(ins, k):
    """For the no-shared-memory kernels, decide whether an ld.global reads the A
    or the B operand by looking back for the c[0x1][..] address add: c[0x1][0x0]
    = A (row-contiguous reuse across k -> L1/L2), c[0x1][0x10] = B (new line each k)."""
    m = ADDR.search(ins[k]["srcs"])
    base = m.group(1) if m else None
    for j in range(k - 1, max(-1, k - 12), -1):
        s = ins[j]["srcs"] + " " + ins[j]["dst"]
        if base and ins[j]["dst"].split()[0] == base:
            if "c[0x1][0x0]" in s:
                return "A"
            if "c[0x1][0x10]" in s:
                return "B"
            if "0x" in s and "c[" not in s:
                # r72 = r71 + 0x2000 style: still the same operand as the chain root
                continue
    return "B"


def simulate(k, args):
    cfg = KERNELS[k]
    ins = load_ir(cfg["log"], cfg["blocks"])
    kinds = [classify(i) for i in ins]
    # bank conflicts from SASS (with .reuse) aligned to IR ffmas
    sass = [x for x in loopstats.parse(cfg["sass"])]
    ranges = []
    for r in cfg["ranges"].split(","):
        lo, hi = r.split("-")
        ranges.append((int(lo, 16), int(hi, 16)))
    sass_ffma = [x for x in sass if loopstats.in_ranges(x[0], ranges) and x[2] == "FFMA"]
    ir_ffma_idx = [n for n, kd in enumerate(kinds) if kd == "ffma"]
    conflict = [False] * len(ins)
    if len(sass_ffma) == len(ir_ffma_idx) and not args.no_conflicts:
        for n, x in zip(ir_ffma_idx, sass_ffma):
            conflict[n] = loopstats.ffma_bank_conflict(x[4])[0]
    elif not args.no_conflicts:
        print(f"warning: ffma count mismatch sass={len(sass_ffma)} ir={len(ir_ffma_idx)}; conflicts ignored")
    # per-instruction memory cost / latency
    cost = [0] * len(ins)
    lat = [0] * len(ins)
    lines = [0] * len(ins)
    lanes = args.lanes or cfg["lanes"]
    for n, i in enumerate(ins):
        kd = kinds[n]
        if kd in ("lds", "sts"):
            m = ADDR.search(i["srcs"])
            base, imm = (m.group(1), int(m.group(2), 16) if m.group(2) else 0) if m else ("", 0)
            if kd == "lds":
                cost[n] = 0 if args.lds_free else max(args.lds_min, (cfg["lds"](base, imm) if cfg["lds"] else 4))
            else:
                cost[n] = 0 if args.lds_free else 4
            lat[n] = args.lds_base
        elif kd in ("ldg", "stg"):
            cost[n] = 0 if args.mem_free else max(1, lanes // 8)
            if cfg["ldgclass"] == "by_cbuf":
                cls = ldg_class_by_cbuf(ins, n)
                if args.show_ldg:
                    print(f"  ldg #{n} {i['dst']} {i['srcs'][:30]} -> class {cls}")
                lat[n] = 0 if args.mem_free else (args.L_A if cls == "A" else args.L_dram)
                # A: row-contiguous reuse across k (L1/L2 resident) -> no DRAM lines;
                # B: distinct float4 per tid.x lane -> (lanes/ty_groups)*16 B contiguous
                lines[n] = 0 if cls == "A" else max(1, (lanes // cfg.get("ty_per_warp", 1)) * 16 // 128)
            else:
                lat[n] = 0 if args.mem_free else args.L_dram
                lines[n] = max(1, lanes * 16 // 128)
            if args.mem_free:
                lines[n] = 0
    delays = [1 if args.no_delays else max(1, i["delay"]) for i in ins]
    if args.ffma_only:
        keep = [n for n in range(len(ins)) if kinds[n] == "ffma"]
        ins = [ins[n] for n in keep]
        kinds = [kinds[n] for n in keep]
        conflict = [conflict[n] for n in keep]
        cost = [cost[n] for n in keep]
        lat = [lat[n] for n in keep]
        lines = [lines[n] for n in keep]
        delays = [delays[n] for n in keep]

    wpw = args.warps_per_wg or cfg["warps_per_wg"]
    wgs = args.wgs or cfg["wgs"]
    nw = wpw * wgs
    N = len(ins)
    # warp state
    pc = [0] * nw
    ready_at = [0] * nw
    sb = [[0] * 6 for _ in range(nw)]
    sbk = [["none"] * 6 for _ in range(nw)]
    sbrd = [[False] * 6 for _ in range(nw)]
    iters = [0] * nw
    at_bar = [False] * nw
    bar_count = collections.Counter()
    # start WGs with a stagger so phase is not forced; 0 = all together
    for w in range(nw):
        ready_at[w] = (w // wpw) * args.stagger
    # pipes: shared memory and LSU, each a single server (busy_until)
    shm_free = 0
    lsu_free = 0
    l2_free = 0
    dram_free = 0
    busy = collections.Counter()
    why = collections.Counter()
    pending = []  # (complete_time, warp, sb_index)
    import heapq
    ffma_issued = 0
    ffma_hist = []
    conflict_stalls = 0
    sched_busy = [0] * 4
    t = 0
    stats_from = None
    ffma_at_start = 0
    done_iters_target = args.iters
    rr = [0] * 4
    max_t = 5_000_000
    while t < max_t:
        # retire completed memory ops
        while pending and pending[0][0] <= t:
            _, w, s = heapq.heappop(pending)
            sb[w][s] -= 1
        for s in range(4):
            ws = [w for w in range(nw) if w % 4 == s]
            n = len(ws)
            issued = False
            reasons = set()
            for q in range(n):
                w = ws[(rr[s] + q) % n]
                if iters[w] >= done_iters_target:
                    continue
                if ready_at[w] > t:
                    reasons.add("delay")
                    continue
                i = ins[pc[w]]
                kd = kinds[pc[w]]
                if any(sb[w][x] > 0 and not (args.relax_rd and sbrd[w][x]) for x in i["wt"]):
                    reasons.add("sb_" + ("ldg" if any(sbk[w][x] == "ldg" for x in i["wt"]) else "shm"))
                    continue
                if kd == "bar":
                    reasons.add("bar")
                    if not at_bar[w]:
                        at_bar[w] = True
                        bar_count[w // wpw] += 1
                    if bar_count[w // wpw] < wpw:
                        continue
                    # all arrived: release everyone in this WG
                    # (each warp retires its own bar when it gets here)
                    at_bar[w] = False
                    # decrement when the last warp passes
                    bar_pass = bar_count.get(("pass", w // wpw), 0) + 1
                    bar_count[("pass", w // wpw)] = bar_pass
                    if bar_pass == wpw:
                        bar_count[w // wpw] = 0
                        bar_count[("pass", w // wpw)] = 0
                # issue
                d = delays[pc[w]]
                if kd == "ffma":
                    ffma_issued += 1
                    if conflict[pc[w]]:
                        d += 1
                        conflict_stalls += 1
                if kd in ("lds", "sts"):
                    if i["wr"] is not None:
                        sbk[w][i["wr"]] = "shm"
                    if i["rd"] is not None:
                        sbk[w][i["rd"]] = "shm"
                    start = max(t + 1, shm_free)
                    shm_free = start + cost[pc[w]]
                    busy["shm"] += cost[pc[w]]
                    if i["rd"] is not None:
                        sb[w][i["rd"]] += 1
                        sbrd[w][i["rd"]] = True
                        heapq.heappush(pending, (start, w, i["rd"]))
                    if i["wr"] is not None:
                        sb[w][i["wr"]] += 1
                        sbrd[w][i["wr"]] = False
                        heapq.heappush(pending, (shm_free + lat[pc[w]], w, i["wr"]))
                elif kd in ("ldg", "stg"):
                    if i["wr"] is not None:
                        sbk[w][i["wr"]] = "ldg"
                    if i["rd"] is not None:
                        sbk[w][i["rd"]] = "ldg"
                    start = max(t + 1, lsu_free)
                    lsu_free = start + cost[pc[w]]
                    busy["lsu"] += cost[pc[w]]
                    done = lsu_free
                    if args.l2_bpc and lines[pc[w]]:
                        l2start = max(done, l2_free)
                        l2_free = l2start + lines[pc[w]] * 128 // args.l2_bpc
                        done = l2_free
                    if args.dram_bpc and lines[pc[w]] and lat[pc[w]] == args.L_dram and not args.mem_free:
                        dstart = max(done, dram_free)
                        dram_free = dstart + lines[pc[w]] * 128 // args.dram_bpc
                        busy["dram"] += dram_free - dstart
                        done = dram_free
                    if i["rd"] is not None:
                        sb[w][i["rd"]] += 1
                        sbrd[w][i["rd"]] = True
                        heapq.heappush(pending, (start + args.rd_extra, w, i["rd"]))
                    if i["wr"] is not None:
                        sb[w][i["wr"]] += 1
                        sbrd[w][i["wr"]] = False
                        heapq.heappush(pending, (done + lat[pc[w]], w, i["wr"]))
                else:
                    if i["wr"] is not None:   # e.g. s2r; treat as short
                        sb[w][i["wr"]] += 1
                        heapq.heappush(pending, (t + 20, w, i["wr"]))
                    if i["rd"] is not None:
                        sb[w][i["rd"]] += 1
                        heapq.heappush(pending, (t + 2, w, i["rd"]))
                ready_at[w] = t + d
                sched_busy[s] += 1
                pc[w] += 1
                if pc[w] == N:
                    pc[w] = 0
                    iters[w] += 1
                rr[s] = (rr[s] + q + 1) % n
                issued = True
                busy["issue"] += 1
                if kd == "ffma":
                    busy["issue_ffma"] += 1
                break
            if not issued and stats_from is not None:
                if "delay" in reasons and len(reasons) == 1:
                    why["delay"] += 1
                elif reasons:
                    why["+".join(sorted(reasons))] += 1
                else:
                    why["bar/none"] += 1
        if stats_from is None and min(iters) >= args.warmup:
            stats_from = t
            ffma_at_start = ffma_issued
            busy.clear()
        if min(iters) >= done_iters_target:
            break
        t += 1
    cycles = t - stats_from
    ffma = ffma_issued - ffma_at_start
    util = ffma * (lanes / 32.0) / (4.0 * cycles)
    per_iter = cycles / (done_iters_target - args.warmup)
    return dict(cycles=cycles, ffma=ffma, util=util, gflops=util * 235.9, per_iter=per_iter,
                conflict_stalls=conflict_stalls, nw=nw, n_ins=N,
                shm=busy["shm"] / cycles, lsu=busy["lsu"] / cycles, dram=busy["dram"] / cycles,
                issue=busy["issue"] / (4.0 * cycles), why={k: round(v / (4.0 * cycles), 3) for k, v in why.most_common(6)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("--L_dram", type=int, default=500)
    ap.add_argument("--L_A", type=int, default=250, help="latency of A-operand loads in no-shared kernels")
    ap.add_argument("--lds_base", type=int, default=20)
    ap.add_argument("--lds_min", type=int, default=0, help="floor on bank-cycles per LDS (4 = no broadcast benefit)")
    ap.add_argument("--show-ldg", action="store_true")
    ap.add_argument("--relax-rd", action="store_true", help="ignore waits on read-scoreboards of memory ops")
    ap.add_argument("--rd_extra", type=int, default=0, help="extra cycles before a memory op's rd scoreboard clears")
    ap.add_argument("--l2_bpc", type=int, default=0, help="L2->SM bytes/clk cap (0 = unlimited)")
    ap.add_argument("--dram_bpc", type=int, default=0, help="effective DRAM bytes/clk for DRAM-class loads (0 = unlimited)")
    ap.add_argument("--no-delays", action="store_true")
    ap.add_argument("--lds-free", action="store_true")
    ap.add_argument("--mem-free", action="store_true")
    ap.add_argument("--no-conflicts", action="store_true")
    ap.add_argument("--ffma-only", action="store_true")
    ap.add_argument("--warps-per-wg", type=int, default=0)
    ap.add_argument("--wgs", type=int, default=0)
    ap.add_argument("--lanes", type=int, default=0)
    ap.add_argument("--stagger", type=int, default=0)
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=6)
    args = ap.parse_args()
    r = simulate(args.kernel, args)
    print(f"{args.kernel}: warps={r['nw']} ins/iter={r['n_ins']} cycles/iter(SM)={r['per_iter']:.0f} "
          f"FFMA util={r['util']*100:.1f}% -> {r['gflops']:.1f} GFLOPS | issue={r['issue']*100:.0f}% "
          f"shm-pipe={r['shm']*100:.0f}% lsu={r['lsu']*100:.0f}% dram={r['dram']*100:.0f}% idle-reasons={r['why']}")


if __name__ == "__main__":
    main()
