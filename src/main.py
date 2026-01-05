import csv
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import deque

@dataclass
class Proc:
    pid: str
    arrival: int
    burst: int
    prio: int
    remaining: int
    start: Optional[int] = None
    finish: Optional[int] = None

def read_csv(path: str) -> List[Proc]:
    pr_map = {"high": 0, "normal": 1, "low": 2}
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row["Process_ID"].strip()
            arrival = int(row["Arrival_Time"])
            burst = int(row["CPU_Burst_Time"])
            pr = pr_map[row["Priority"].strip().lower()]
            out.append(Proc(pid=pid, arrival=arrival, burst=burst, prio=pr, remaining=burst))
    return out

def clone_procs(procs: List[Proc]) -> List[Proc]:
    return [Proc(p.pid, p.arrival, p.burst, p.prio, p.burst) for p in procs]

def compress_gantt(raw: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    if not raw:
        return []
    out = [raw[0]]
    for s, e, pid in raw[1:]:
        ps, pe, ppid = out[-1]
        if pid == ppid and s == pe:
            out[-1] = (ps, e, ppid)
        else:
            out.append((s, e, pid))
    return out

def context_switches_from_gantt(gantt: List[Tuple[int, int, str]]) -> int:
    cnt = 0
    prev = None
    for _, _, pid in gantt:
        if prev is None:
            prev = pid
            continue
        if pid != prev and pid != "IDLE" and prev != "IDLE":
            cnt += 1
        prev = pid
    return cnt

def idle_time_from_gantt(gantt: List[Tuple[int, int, str]]) -> int:
    return sum(e - s for s, e, pid in gantt if pid == "IDLE")

def compute_metrics(procs: List[Proc], gantt: List[Tuple[int, int, str]], cs: int) -> Dict[str, float]:
    waits = []
    tats = []
    for p in procs:
        tat = p.finish - p.arrival
        wt = tat - p.burst
        tats.append(tat)
        waits.append(wt)
    total_time = gantt[-1][1] if gantt else 0
    idle = idle_time_from_gantt(gantt)
    overhead = cs * 0.001
    eff = 0.0
    if total_time > 0:
        eff = (total_time - idle - overhead) / total_time
    return {
        "avg_waiting": sum(waits) / len(waits) if waits else 0.0,
        "max_waiting": max(waits) if waits else 0.0,
        "avg_turnaround": sum(tats) / len(tats) if tats else 0.0,
        "max_turnaround": max(tats) if tats else 0.0,
        "total_time": float(total_time),
        "idle_time": float(idle),
        "context_switches": float(cs),
        "cpu_efficiency": float(eff),
    }

def throughput(procs: List[Proc], T: int) -> float:
    done = sum(1 for p in procs if p.finish is not None and p.finish <= T)
    return done / T

def write_output(path: str, algo_name: str, case_name: str, gantt: List[Tuple[int, int, str]], procs: List[Proc]) -> None:
    cs = context_switches_from_gantt(gantt)
    m = compute_metrics(procs, gantt, cs)
    Ts = [50, 100, 150, 200]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{algo_name} | {case_name}\n\n")
        f.write("CPU Zaman Tablosu\n")
        for s, e, pid in gantt:
            f.write(f"[ {s} ] -- {pid} -- [ {e} ]\n")
        f.write("\nWaiting Time\n")
        f.write(f"Max: {m['max_waiting']}\n")
        f.write(f"Avg: {m['avg_waiting']}\n\n")
        f.write("Turnaround Time\n")
        f.write(f"Max: {m['max_turnaround']}\n")
        f.write(f"Avg: {m['avg_turnaround']}\n\n")
        f.write("Throughput\n")
        for T in Ts:
            f.write(f"T={T}: {throughput(procs, T)}\n")
        f.write("\nCPU Verimliligi\n")
        f.write(f"Baglam Degistirme Suresi: 0.001\n")
        f.write(f"Toplam Baglam Degistirme Sayisi: {int(m['context_switches'])}\n")
        f.write(f"Toplam Sure: {m['total_time']}\n")
        f.write(f"Idle Sure: {m['idle_time']}\n")
        f.write(f"CPU Verimliligi: {m['cpu_efficiency']}\n")

def fcfs(procs: List[Proc]) -> Tuple[List[Tuple[int, int, str]], List[Proc]]:
    procs = sorted(procs, key=lambda x: (x.arrival, x.pid))
    t = 0
    gantt = []
    for p in procs:
        if t < p.arrival:
            gantt.append((t, p.arrival, "IDLE"))
            t = p.arrival
        p.start = t if p.start is None else p.start
        gantt.append((t, t + p.burst, p.pid))
        t += p.burst
        p.finish = t
        p.remaining = 0
    return compress_gantt(gantt), procs

def sjf_nonpreemptive(procs: List[Proc]) -> Tuple[List[Tuple[int, int, str]], List[Proc]]:
    procs = sorted(procs, key=lambda x: (x.arrival, x.pid))
    n = len(procs)
    i = 0
    t = 0
    ready = []
    gantt = []
    done = 0
    while done < n:
        while i < n and procs[i].arrival <= t:
            ready.append(procs[i])
            i += 1
        if not ready:
            if i < n:
                if t < procs[i].arrival:
                    gantt.append((t, procs[i].arrival, "IDLE"))
                    t = procs[i].arrival
                continue
            break
        ready.sort(key=lambda x: (x.burst, x.arrival, x.pid))
        p = ready.pop(0)
        p.start = t if p.start is None else p.start
        gantt.append((t, t + p.remaining, p.pid))
        t += p.remaining
        p.remaining = 0
        p.finish = t
        done += 1
    return compress_gantt(gantt), sorted(procs, key=lambda x: x.pid)

def sjf_preemptive(procs: List[Proc]) -> Tuple[List[Tuple[int, int, str]], List[Proc]]:
    procs = sorted(procs, key=lambda x: (x.arrival, x.pid))
    n = len(procs)
    i = 0
    t = 0
    ready = []
    gantt = []
    finished = 0
    current = None
    while finished < n:
        while i < n and procs[i].arrival == t:
            ready.append(procs[i])
            i += 1
        candidates = ready[:]
        if current is not None and current.remaining > 0:
            candidates.append(current)
        if not candidates:
            next_arr = procs[i].arrival if i < n else None
            if next_arr is None:
                break
            if t < next_arr:
                gantt.append((t, next_arr, "IDLE"))
                t = next_arr
            continue
        candidates.sort(key=lambda x: (x.remaining, x.arrival, x.pid))
        chosen = candidates[0]
        if current is not None and current is not chosen and current.remaining > 0:
            ready.append(current)
        if chosen in ready:
            ready.remove(chosen)
        current = chosen
        if current.start is None:
            current.start = t
        gantt.append((t, t + 1, current.pid))
        current.remaining -= 1
        t += 1
        if current.remaining == 0:
            current.finish = t
            finished += 1
            current = None
    return compress_gantt(gantt), sorted(procs, key=lambda x: x.pid)

def priority_nonpreemptive(procs: List[Proc]) -> Tuple[List[Tuple[int, int, str]], List[Proc]]:
    procs = sorted(procs, key=lambda x: (x.arrival, x.pid))
    n = len(procs)
    i = 0
    t = 0
    ready = []
    gantt = []
    done = 0
    while done < n:
        while i < n and procs[i].arrival <= t:
            ready.append(procs[i])
            i += 1
        if not ready:
            if i < n:
                if t < procs[i].arrival:
                    gantt.append((t, procs[i].arrival, "IDLE"))
                    t = procs[i].arrival
                continue
            break
        ready.sort(key=lambda x: (x.prio, x.arrival, x.pid))
        p = ready.pop(0)
        p.start = t if p.start is None else p.start
        gantt.append((t, t + p.remaining, p.pid))
        t += p.remaining
        p.remaining = 0
        p.finish = t
        done += 1
    return compress_gantt(gantt), sorted(procs, key=lambda x: x.pid)

def priority_preemptive(procs: List[Proc]) -> Tuple[List[Tuple[int, int, str]], List[Proc]]:
    procs = sorted(procs, key=lambda x: (x.arrival, x.pid))
    n = len(procs)
    i = 0
    t = 0
    ready = []
    gantt = []
    finished = 0
    current = None
    while finished < n:
        while i < n and procs[i].arrival == t:
            ready.append(procs[i])
            i += 1
        candidates = ready[:]
        if current is not None and current.remaining > 0:
            candidates.append(current)
        if not candidates:
            next_arr = procs[i].arrival if i < n else None
            if next_arr is None:
                break
            if t < next_arr:
                gantt.append((t, next_arr, "IDLE"))
                t = next_arr
            continue
        candidates.sort(key=lambda x: (x.prio, x.arrival, x.pid))
        chosen = candidates[0]
        if current is not None and current is not chosen and current.remaining > 0:
            ready.append(current)
        if chosen in ready:
            ready.remove(chosen)
        current = chosen
        if current.start is None:
            current.start = t
        gantt.append((t, t + 1, current.pid))
        current.remaining -= 1
        t += 1
        if current.remaining == 0:
            current.finish = t
            finished += 1
            current = None
    return compress_gantt(gantt), sorted(procs, key=lambda x: x.pid)

def round_robin(procs: List[Proc], quantum: int = 4) -> Tuple[List[Tuple[int, int, str]], List[Proc]]:
    procs = sorted(procs, key=lambda x: (x.arrival, x.pid))
    n = len(procs)
    i = 0
    t = 0
    q = deque()
    gantt = []
    finished = 0
    current = None
    slice_left = 0
    while finished < n:
        while i < n and procs[i].arrival == t:
            q.append(procs[i])
            i += 1
        if current is None:
            if q:
                current = q.popleft()
                if current.start is None:
                    current.start = t
                slice_left = quantum
            else:
                next_arr = procs[i].arrival if i < n else None
                if next_arr is None:
                    break
                if t < next_arr:
                    gantt.append((t, next_arr, "IDLE"))
                    t = next_arr
                continue
        gantt.append((t, t + 1, current.pid))
        current.remaining -= 1
        slice_left -= 1
        t += 1
        while i < n and procs[i].arrival == t:
            q.append(procs[i])
            i += 1
        if current.remaining == 0:
            current.finish = t
            finished += 1
            current = None
            slice_left = 0
        elif slice_left == 0:
            q.append(current)
            current = None
    return compress_gantt(gantt), sorted(procs, key=lambda x: x.pid)

def run_all(case_path: str, out_dir: str, quantum: int) -> None:
    base = os.path.splitext(os.path.basename(case_path))[0]
    original = read_csv(case_path)
    algos = [
        ("FCFS", lambda ps: fcfs(ps)),
        ("SJF_Preemptive", lambda ps: sjf_preemptive(ps)),
        ("SJF_NonPreemptive", lambda ps: sjf_nonpreemptive(ps)),
        ("RoundRobin", lambda ps: round_robin(ps, quantum=quantum)),
        ("Priority_Preemptive", lambda ps: priority_preemptive(ps)),
        ("Priority_NonPreemptive", lambda ps: priority_nonpreemptive(ps)),
    ]
    os.makedirs(out_dir, exist_ok=True)
    for name, fn in algos:
        ps = clone_procs(original)
        gantt, done = fn(ps)
        out_path = os.path.join(out_dir, f"{base}_{name}.txt")
        write_output(out_path, name, base, gantt, done)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, default="")
    ap.add_argument("--out", type=str, default="outputs")
    ap.add_argument("--quantum", type=int, default=4)
    args = ap.parse_args()
    if args.case:
        run_all(args.case, args.out, args.quantum)
    else:
        run_all(os.path.join("data", "case1.csv"), args.out, args.quantum)
        run_all(os.path.join("data", "case2.csv"), args.out, args.quantum)

if __name__ == "__main__":
    main()
