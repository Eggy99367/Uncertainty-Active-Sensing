#!/usr/bin/env python3
"""ESP32 CSI + LD2450 synchronized live acquisition and posture experiment."""
import argparse
import csv
import math
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation

STAGES = [("EMPTY", "EMPTY"), ("SITTING", "SITTING"), ("LYING", "LYING")]
ROUNDS = 10
PREP_SECONDS = 15
RECORD_SECONDS = 30
BLOCK_SECONDS = PREP_SECONDS + RECORD_SECONDS
TOTAL_SECONDS = ROUNDS * len(STAGES) * BLOCK_SECONDS
MAX_DISPLAY = 500
PRESENCE_ON_SECONDS = 0.8
PRESENCE_OFF_SECONDS = 1.5
BED_ROI = (-1000, 1000, 600, 2100)
VOICE_ENABLED = True
VOICE = "Samantha"
VOICE_RATE = 200
SPOKEN_NAMES = {"EMPTY": "empty", "SITTING": "sitting", "LYING": "lying"}


def play_sound(kind):
    """macOS system sounds, terminal bell fallback."""
    sound = {"prepare": "Pop", "tick": "Tink", "start": "Glass", "end": "Hero", "finish": "Funk"}[kind]
    path = Path(f"/System/Library/Sounds/{sound}.aiff")
    if sys.platform == "darwin" and path.exists():
        try:
            subprocess.Popen(["afplay", str(path)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    print("\a", end="", flush=True)


_speech = [None]


def speak(text, fallback, interrupt=True):
    if VOICE_ENABLED and sys.platform == "darwin":
        try:
            if _speech[0] is not None and _speech[0].poll() is None:
                if interrupt:
                    _speech[0].terminate()
                else:
                    _speech[0].wait(timeout=10)
            _speech[0] = subprocess.Popen(["say", "-v", VOICE, "-r", str(VOICE_RATE), text],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    play_sound(fallback)


def stage_at(elapsed):
    if elapsed < 0:
        return "WAITING", "WAITING", -1, 0, 0.0
    if elapsed >= TOTAL_SECONDS:
        return "DONE", "DONE", -1, 0, 0.0
    idx = int(elapsed // BLOCK_SECONDS)
    local = elapsed - idx * BLOCK_SECONDS
    label, name = STAGES[idx % len(STAGES)]
    rnd = idx // len(STAGES) + 1
    if local < PREP_SECONDS:
        return "TRANSITION", name, idx, rnd, PREP_SECONDS - local
    return label, name, idx, rnd, local - PREP_SECONDS


def in_bed_roi(x, y):
    """Return whether a radar target lies within the configured bed ROI."""
    return BED_ROI is None or (BED_ROI[0] <= x <= BED_ROI[1] and
                               BED_ROI[2] <= y <= BED_ROI[3])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/cu.usbserial-0001")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--output", default="experiment_results")
    args = parser.parse_args()

    folder = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    raw_path = folder / "all_packets.csv"
    csi_path = folder / "csi_features.csv"
    radar_path = folder / "radar_targets.csv"
    summary_path = folder / "stage_summary.csv"
    presence_path = folder / "radar_presence.csv"

    lock = threading.Lock()
    stop = threading.Event()
    t0 = [None]
    data = {"t": deque(maxlen=MAX_DISPLAY), "amp": deque(maxlen=MAX_DISPLAY),
            "variance": deque(maxlen=MAX_DISPLAY), "rssi": deque(maxlen=MAX_DISPLAY),
            "radar_t": deque(maxlen=MAX_DISPLAY), "radar_dist": deque(maxlen=MAX_DISPLAY),
            "radar_speed": deque(maxlen=MAX_DISPLAY),
            "presence_t": deque(maxlen=MAX_DISPLAY),
            "presence": deque(maxlen=MAX_DISPLAY)}
    amp_window = deque(maxlen=20)
    counts = {"CSI": 0, "RADAR": 0, "BAD": 0}
    all_csi = []
    all_radar = []
    messages = queue.Queue()
    state = {"serial": "Connecting", "experiment": "WAITING", "presence": False}
    # All three target slots share one ESP timestamp; evaluate them together.
    tracker = {"candidate_since": None, "absent_since": None, "present": False,
               "frame_ms": None, "frame_targets": [], "last_frame_host": None,
               "last_frame_elapsed": None, "last_frame_stage": None,
               "last_frame_round": None}

    def reader(ser):
        with raw_path.open("w", newline="", buffering=1) as fraw, \
             csi_path.open("w", newline="", buffering=1) as fcsi, \
             radar_path.open("w", newline="", buffering=1) as fradar, \
             presence_path.open("w", newline="", buffering=1) as fpresence:
            wr = csv.writer(fraw)
            wc = csv.writer(fcsi)
            wm = csv.writer(fradar)
            wp = csv.writer(fpresence)
            wr.writerow(["host_iso", "elapsed_s", "stage", "round", "raw_line"])
            wc.writerow(["host_iso", "elapsed_s", "stage", "round", "esp_ms", "rssi_dbm",
                         "csi_len", "first_word_invalid", "mean_amplitude", "rolling_variance"])
            wm.writerow(["host_iso", "elapsed_s", "stage", "round", "esp_ms", "target_id",
                         "x_mm", "y_mm", "speed_cm_s", "resolution_mm", "detected", "distance_mm"])
            wp.writerow(["host_iso", "elapsed_s", "stage", "round", "esp_ms",
                         "raw_targets_in_roi", "filtered_human_present"])

            def finish_frame():
                if tracker["frame_ms"] is None:
                    return
                hits = tracker["frame_targets"]
                raw_count = len(hits)
                now_s = tracker["last_frame_elapsed"]
                if raw_count:
                    tracker["absent_since"] = None
                    if tracker["candidate_since"] is None:
                        tracker["candidate_since"] = now_s
                    if not tracker["present"] and now_s - tracker["candidate_since"] >= PRESENCE_ON_SECONDS:
                        tracker["present"] = True
                else:
                    tracker["candidate_since"] = None
                    if tracker["absent_since"] is None:
                        tracker["absent_since"] = now_s
                    if tracker["present"] and now_s - tracker["absent_since"] >= PRESENCE_OFF_SECONDS:
                        tracker["present"] = False
                state["presence"] = tracker["present"]
                wp.writerow([tracker["last_frame_host"], round(now_s, 4),
                             tracker["last_frame_stage"], tracker["last_frame_round"],
                             tracker["frame_ms"], raw_count, int(tracker["present"])])
                with lock:
                    if now_s >= 0:
                        data["presence_t"].append(now_s)
                        data["presence"].append(int(tracker["present"]))
                tracker["frame_targets"] = []
            while not stop.is_set():
                try:
                    incoming = ser.readline()
                except (serial.SerialException, OSError) as exc:
                    messages.put(f"Serial error: {exc}")
                    state["serial"] = "Disconnected"
                    stop.set()
                    break
                if not incoming:
                    continue
                line = incoming.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                now = time.monotonic()
                elapsed = now - t0[0] if t0[0] is not None else -1.0
                stage, _, _, rnd, _ = stage_at(elapsed)
                host = datetime.now().isoformat(timespec="milliseconds")
                wr.writerow([host, round(elapsed, 4), stage, rnd, line])
                if line.startswith("STATUS,"):
                    messages.put(line[:160])
                    continue
                try:
                    fields = next(csv.reader([line]))
                    if fields[0] == "CSI_DATA":
                        if len(fields) != 6:
                            raise ValueError(f"CSI fields={len(fields)}")
                        esp_ms, rssi, n, invalid = map(int, fields[1:5])
                        iq = np.fromstring(fields[5], sep=" ", dtype=np.int16)
                        if len(iq) != n or n % 2 or n < 4:
                            raise ValueError(f"CSI bytes {len(iq)} != {n}")
                        if np.any((iq < -128) | (iq > 127)):
                            raise ValueError("CSI outside int8 range")
                        pairs = iq.reshape(-1, 2).astype(np.float64)
                        if invalid:
                            pairs = pairs[1:]
                        amp = float(np.hypot(pairs[:, 0], pairs[:, 1]).mean())
                        amp_window.append(amp)
                        variance = float(np.var(amp_window))
                        row = [host, round(elapsed, 4), stage, rnd, esp_ms, rssi,
                               n, invalid, amp, variance]
                        wc.writerow(row)
                        with lock:
                            counts["CSI"] += 1
                            if elapsed >= 0:
                                data["t"].append(elapsed)
                                data["amp"].append(amp)
                                data["variance"].append(variance)
                                data["rssi"].append(rssi)
                                all_csi.append(row)
                    elif fields[0] == "RADAR_DATA":
                        if len(fields) != 8:
                            raise ValueError(f"RADAR fields={len(fields)}")
                        esp_ms, target, x, y, speed, resolution, detected = map(int, fields[1:])
                        # Flush previous 3-slot radar frame when timestamp changes.
                        if tracker["frame_ms"] is not None and esp_ms != tracker["frame_ms"]:
                            finish_frame()
                        tracker["frame_ms"] = esp_ms
                        tracker["last_frame_host"] = host
                        tracker["last_frame_elapsed"] = elapsed
                        tracker["last_frame_stage"] = stage
                        tracker["last_frame_round"] = rnd
                        in_roi = in_bed_roi(x, y)
                        if detected and in_roi:
                            tracker["frame_targets"].append(target)
                        dist = math.hypot(x, y) if detected else math.nan
                        row = [host, round(elapsed, 4), stage, rnd, esp_ms, target,
                               x, y, speed, resolution, detected, dist]
                        wm.writerow(row)
                        if target == 3:
                            finish_frame()
                            tracker["frame_ms"] = None
                        with lock:
                            counts["RADAR"] += 1
                            if elapsed >= 0:
                                all_radar.append(row)
                                if target == 1:
                                    data["radar_t"].append(elapsed)
                                    data["radar_dist"].append(dist if in_roi else math.nan)
                                    data["radar_speed"].append(speed if detected and in_roi else math.nan)
                except (ValueError, IndexError, csv.Error) as exc:
                    with lock:
                        counts["BAD"] += 1
                        nbad = counts["BAD"]
                    if nbad <= 12 or nbad % 50 == 0:
                        messages.put(f"Bad packet #{nbad}: {exc} | {line[:70]}")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.25)
    except (serial.SerialException, OSError) as exc:
        print(f"Cannot open {args.port}: {exc}")
        return
    print(f"Connected {args.port} @ {args.baud}")
    print("Close PlatformIO Serial Monitor and other serial readers.")
    print("Allowing ESP32 to reset and reconnect for 3 seconds...")
    time.sleep(3)
    ser.reset_input_buffer()
    state["serial"] = "Connected"
    thread = threading.Thread(target=reader, args=(ser,), daemon=True)
    thread.start()

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("ESP32 CSI + LD2450 | Live posture experiment")
    series = []
    labels = [("CSI mean amplitude", "Amplitude"), ("CSI rolling variance", "Variance"),
              ("Radar target 1 distance (Bed ROI filtered)", "Distance (mm)"),
              ("Radar target 1 speed (Bed ROI filtered)", "Speed (cm/s)"),
              ("Filtered human presence (heuristic)", "Present 0/1")]
    for ax, (title, ylabel) in zip(axes, labels):
        line, = ax.plot([], [], linewidth=1.4)
        series.append(line)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Experiment elapsed time (s)")
    axes[-1].set_ylim(-0.1, 1.1)
    axes[-1].set_yticks([0, 1])
    status = fig.text(0.02, 0.01, "Starting...", fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    last_phase = [None]
    last_tick = [None]
    finalized = [False]

    def finalize():
        if finalized[0]:
            return
        finalized[0] = True
        stop.set()
        thread.join(timeout=2)
        ser.close()
        with lock:
            csi_rows = list(all_csi)
            radar_rows = list(all_radar)
        summaries = []
        for rnd in list(range(1, ROUNDS + 1)) + ["ALL"]:
            for stage_key, stage_name in STAGES:
                cs = [r for r in csi_rows if r[2] == stage_key and (rnd == "ALL" or r[3] == rnd)]
                ca = np.array([r[8] for r in cs], dtype=float)
                cv = np.array([r[9] for r in cs], dtype=float)
                ra = np.array([r[11] for r in radar_rows if r[2] == stage_key and (rnd == "ALL" or r[3] == rnd)
                               and r[5] == 1 and r[10] == 1 and in_bed_roi(r[6], r[7])], dtype=float)
                summaries.append([rnd, stage_key, stage_name, len(ca), float(np.mean(ca)) if len(ca) else math.nan,
                                  float(np.mean(cv)) if len(cv) else math.nan,
                                  len(ra), float(np.mean(ra)) if len(ra) else math.nan])
        with summary_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "stage", "name", "csi_count", "mean_csi_amplitude",
                        "mean_csi_variance", "radar_target1_in_roi_count", "mean_radar_distance_in_roi_mm"])
            w.writerows(summaries)
        fig2, axs = plt.subplots(3, 1, figsize=(12, 11))
        colors = ["tab:green", "tab:blue", "tab:orange"]
        for j, (stage_key, stage_name) in enumerate(STAGES):
            for rnd in range(1, ROUNDS + 1):
                offset = ((rnd - 1) * len(STAGES) + j) * BLOCK_SECONDS + PREP_SECONDS
                label = stage_key if rnd == 1 else None
                ca = [r for r in csi_rows if r[2] == stage_key and r[3] == rnd]
                ra = [r for r in radar_rows if r[2] == stage_key and r[3] == rnd and r[5] == 1]
                if ca:
                    x = np.array([r[1] for r in ca]) - offset
                    axs[0].plot(x, [r[8] for r in ca], label=label, color=colors[j], alpha=0.6)
                    axs[1].plot(x, [r[9] for r in ca], label=label, color=colors[j], alpha=0.6)
                if ra:
                    x = np.array([r[1] for r in ra]) - offset
                    axs[2].plot(x, [r[11] if r[10] == 1 and in_bed_roi(r[6], r[7]) else math.nan for r in ra],
                                label=label, color=colors[j], alpha=0.6)
        for ax, ylabel in zip(axs, ["Mean CSI amplitude", "CSI rolling variance", "Radar target 1 distance (mm)"]):
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, RECORD_SECONDS)
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="upper right")
        axs[-1].set_xlabel("Seconds since observation started")
        fig2.suptitle(f"EMPTY vs SITTING vs LYING ({ROUNDS} rounds, {RECORD_SECONDS}-second observations)\nRadar distance: target 1 inside Bed ROI only")
        fig2.tight_layout()
        comparison = folder / "stage_comparison.png"
        fig2.savefig(comparison, dpi=180)
        plt.close(fig2)
        print(f"\nExperiment complete. CSV and comparison: {folder}", flush=True)
        print(f"Comparison graph: {comparison}", flush=True)
        for r in summaries:
            print(f"Round {r[0]} {r[1]}: CSI={r[3]}, radar target1 ROI detections={r[6]}")
        speak("Experiment complete. Data saved.", "finish", interrupt=False)

    def update(_):
        if t0[0] is None:
            t0[0] = time.monotonic()
            print(f"Experiment started: {ROUNDS} rounds, {PREP_SECONDS}s preparation for EMPTY", flush=True)
        elapsed = time.monotonic() - t0[0]
        phase, name, idx, rnd, phase_time = stage_at(elapsed)
        phase_id = (phase, idx)
        spoken = SPOKEN_NAMES.get(name, name)
        if phase_id != last_phase[0]:
            last_phase[0] = phase_id
            if phase == "TRANSITION":
                print(f"\nRound {rnd}/{ROUNDS} | Prepare {name}: {PREP_SECONDS} seconds", flush=True)
                intro = "Recording finished. " if idx > 0 else ""
                speak(f"{intro}Round {rnd}. Prepare for {spoken}.", "prepare")
            elif phase in [s[0] for s in STAGES]:
                print(f"\nRound {rnd}/{ROUNDS} | RECORDING {name}: {RECORD_SECONDS} seconds", flush=True)
                speak(f"Start recording {spoken}.", "start")
            elif phase == "DONE":
                print("\nFinal recording finished", flush=True)
                speak("All recordings finished. Saving data.", "end")
                finalize()
        if phase == "TRANSITION":
            remaining = int(math.ceil(phase_time))
            tick_id = (idx, remaining)
            if remaining in (3, 2, 1) and tick_id != last_tick[0]:
                last_tick[0] = tick_id
                print(f"{name} starts in {remaining}...", flush=True)
                play_sound("tick")
        if phase in [s[0] for s in STAGES] and phase_time >= RECORD_SECONDS - 0.2:
            pass
        with lock:
            t = list(data["t"])
            amp = list(data["amp"])
            var = list(data["variance"])
            rt = list(data["radar_t"])
            rd = list(data["radar_dist"])
            rs = list(data["radar_speed"])
            count = dict(counts)
        with lock:
            pt = list(data["presence_t"])
            pv = list(data["presence"])
        for line, xs, ys, ax in zip(series,
                                   [t, t, rt, rt, pt], [amp, var, rd, rs, pv], axes):
            line.set_data(xs, ys)
            if xs:
                ax.relim()
                if ax is not axes[-1]:
                    ax.autoscale_view(scalex=False, scaley=True)
        for ax in axes:
            ax.set_xlim(max(0, elapsed - 45), max(45, elapsed + 1))
        status.set_text(f"Round {rnd}/{ROUNDS} | {phase}: {name} | {phase_time:.1f}s | CSI {count['CSI']} | "
                        f"RADAR {count['RADAR']} | HUMAN {'YES' if state['presence'] else 'NO'} "
                        f"(heuristic) | BAD {count['BAD']} | {state['serial']}")
        while not messages.empty():
            print(messages.get_nowait(), flush=True)
        return (*series, status)

    ani = FuncAnimation(fig, update, interval=150, blit=False, cache_frame_data=False)

    def on_close(_):
        if not finalized[0]:
            print("Window closed early: saving partial data and comparison.", flush=True)
            finalize()

    fig.canvas.mpl_connect("close_event", on_close)
    try:
        plt.show()
    finally:
        finalize()


if __name__ == "__main__":
    main()