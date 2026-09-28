"""Recompute the README's headline numbers from the logs and metadata shipped in records/.

Reads files only; launches no training or evaluation. Exits 1 if any check fails.
Run from the repository root: python3 scripts/check_readme_claims.py
"""

import json
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
R10 = ROOT / "records/track_10min_16mb"
RNR = ROOT / "records/track_non_record_16mb"
CAP_BYTES = 16_000_000

results = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))


def last(path, pattern):
    hits = re.findall(pattern, path.read_text(errors="replace"))
    return hits[-1] if hits else None


def floats(paths, pattern):
    return [float(last(p, pattern)) for p in paths]


def record_track(label, directory, seeds, claim, tokens_pattern, tokens):
    logs = [directory / f"train_seed{s}.log" for s in seeds]
    bpb = floats(logs, r"quantized_ttt_phased val_loss:[\d.]+ val_bpb:([\d.]+)")
    sizes = [int(last(p, r"Total submission size quantized\+pergroup: (\d+) bytes")) for p in logs]
    train_ms = [int(last(p, r"stopping_early: wallclock_cap train_time: ?(\d+)ms")) for p in logs]
    eval_s = floats(logs, r"total_eval_time:([\d.]+)s")
    counts = {int(last(p, tokens_pattern)) for p in logs}
    m = mean(bpb)
    check(f"{label}: three-seed mean BPB", round(m, 5) == claim, f"{m:.8f} from {len(logs)} logs; README {claim}")
    check(f"{label}: artifacts under 16,000,000 bytes", max(sizes) < CAP_BYTES, f"max {max(sizes):,}")
    check(f"{label}: training under 600 s", max(train_ms) < 600_000, f"max {max(train_ms):,} ms")
    check(f"{label}: evaluation under 600 s", max(eval_s) < 600, f"max {max(eval_s)} s")
    check(f"{label}: scored targets", counts == {tokens}, f"{sorted(counts)}; README {tokens:,}")
    return m


winner = record_track("1st codemath3000", R10 / "2026-05-01_SP8192_PR2130Base_Calib32", [314, 42, 0], 1.05651, r"val_tokens: (\d+)", 47_851_520)
second = record_track("2nd simonbissonnette", R10 / "2026-04-30_SP8192_CaseOps_Progressive3k_ShortDocTTT", [42, 314, 0], 1.05759, r"target_tokens:(\d+)", 47_853_343)
record_track("3rd andrewbaggio1", R10 / "2026-04-30_LongCtx_NoQV_QK525_on_1945_1.0586", [42, 0, 1234], 1.05855, r"val_tokens: (\d+)", 47_851_520)

base_log = R10 / "2026-03-17_NaiveBaseline/train.log"
base = float(last(base_log, r"final_int8_zlib_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)"))
check("Baseline: final int8+zlib BPB", round(base, 4) == 1.2244, f"{base:.8f}")
drop = (1.2244 - 1.0565) / 1.2244
gap = (1.0576 - 1.0565) / 1.0576
check("Winner vs baseline, rounded leaderboard values", round(drop * 100, 1) == 13.7, f"{drop:.2%} lower")
check("First vs second, rounded leaderboard values", round(gap * 100, 2) == 0.10, f"{gap:.3%}")

ternary = RNR.parent / "track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon"
t = floats(sorted(ternary.glob("ternary_log_*.txt")), r"final_sliding val_loss:[\d.]+ val_bpb:([\d.]+)")
check("Ternary: three-seed sliding mean", len(t) == 3 and round(mean(t), 4) == 1.1570, f"{mean(t):.5f} from {t}")

binary = RNR / "2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear/binary_log.txt"
b = float(last(binary, r"final_sliding val_loss:[\d.]+ val_bpb:([\d.]+)"))
b_hours = int(last(binary, r"train_time:(\d+)ms")) / 3_600_000
check("Binary: sliding BPB", b == 1.1239, f"{b}")
check("Binary: training time, about 2.15 hours", abs(b_hours - 2.15) < 0.02, f"{b_hours:.3f} h")

jepa = RNR / "2026-03-26_37M_LeWM_Jepa_Mamba2_10L_UNet_INT4FP8QAT_Brotli"
j_long = float(last(jepa / "train_log_bpe_100k_steps.txt", r"final_sliding val_loss:[\d.]+ val_bpb:([\d.]+)"))
j_10 = float(last(jepa / "train_log_bpe_10min.txt", r"final_sliding val_loss:[\d.]+ val_bpb:([\d.]+)"))
check("JEPA: 100k-step BPE sliding BPB", j_long == 1.2064, f"{j_long}")
check("JEPA: 10-minute BPE sliding BPB", j_10 == 1.2566, f"{j_10}")

hnet = RNR / "2026-03-29_HNet_ByteVsSubword_Study"
exact = r"final_int8_zlib_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)"
h4 = float(last(hnet / "train_byte260_4h.log", exact))
h10 = float(last(hnet / "train_byte260_best_10_min.log", exact))
check("H-Net: 4-hour byte BPB", round(h4, 4) == 1.3595, f"{h4:.8f}")
check("H-Net: shipped 10-minute byte log (one seed, not the 1.4116 mean)", round(h10, 4) == 1.4032, f"{h10:.8f}")

mamba = float(last(RNR / "2026-04-15_Mamba3Hybrid_SP8192_GPTQ_TTT/train_seed1337.log", exact))
check("Mamba-3 hybrid: final BPB", round(mamba, 4) == 1.1473, f"{mamba:.8f}")

uni = RNR / "2026-03-29_Universal_Transformer"
u = float(last(uni / "train_seed42.txt", r"final_int6_zstd_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)"))
u_bytes = json.loads((uni / "submission.json").read_text())["bytes_total"]
check("Universal Transformer: final BPB", round(u, 4) == 1.2249, f"{u:.8f}")
check("Universal Transformer: 4.95 MB artifact", round(u_bytes / 1e6, 2) == 4.95, f"{u_bytes:,} bytes")

leg = float(last(RNR / "2026-03-31_LegendreGPT/train.log", r"final_INT7_zlib_roundtrip val_loss:[\d.]+ val_bpb:([\d.]+)"))
check("LegendreGPT: shipped log is INT7+zlib, not the 1.2266 post-hoc export", leg == 1.2353, f"{leg}")

adapter = sorted((RNR / "2026-04-30_Random_Linear_Adapter").glob("train_seed_*.txt"))
a = floats(adapter, r"final_int8_zlib_roundtrip_exact sliding val_loss:[\d.]+ sliding val_bpb:([\d.]+)")
check("Random adapters: listed 1.1971 is the worst of three seeds", len(a) == 3 and round(max(a), 4) == 1.1971, f"{[round(x, 4) for x in a]}, mean {mean(a):.4f}")

dg = RNR / "2026-03-23_DGAttention_DavidGao/train.log"
dg_text = dg.read_text(errors="replace")
dg_last = float(last(dg, r"step:\d+/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)"))
dg_bytes = int(last(dg, r"Total submission size: (\d+) bytes"))
check("DG Attention: 1.1898 is the last in-training (pre-export) validation", dg_last == 1.1898, f"{dg_last}")
check("DG Attention: log has no post-quantization roundtrip score", "roundtrip" not in dg_text, "no 'roundtrip' line")
check("DG Attention: artifact exceeds the 16,000,000-byte cap", dg_bytes > CAP_BYTES, f"{dg_bytes:,} bytes")

mdlm = RNR / "2026-03-29_LLaDA_MDLM_Diffusion"
mdlm_logs = [p for p in mdlm.iterdir() if p.suffix in {".log", ".txt"}]
mdlm_meta = json.loads((mdlm / "submission.json").read_text())
check("MDLM: no training log shipped, so 1.1465 cannot be checked on disk", not mdlm_logs, f"{len(mdlm_logs)} log files")
check("MDLM: no artifact byte count in metadata", "bytes_total" not in mdlm_meta, "bytes_total absent")

width = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
failed = sum(not ok for _, ok, _ in results)
print(f"\n{len(results) - failed} of {len(results)} checks pass")
sys.exit(1 if failed else 0)
