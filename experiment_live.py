#!/usr/bin/env python3
"""Record timestamped ESP32 telemetry from the serial port."""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import serial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/cu.usbserial-0001")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--output", default="experiment_results")
    args = parser.parse_args()
    folder = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with serial.Serial(args.port, args.baud, timeout=0.25) as ser:
            time.sleep(3)
            ser.reset_input_buffer()
            start = time.monotonic()
            with (folder / "all_packets.csv").open("w", newline="", buffering=1) as output:
                writer = csv.writer(output)
                writer.writerow(["host_iso", "elapsed_s", "raw_line"])
                while True:
                    incoming = ser.readline()
                    if incoming:
                        line = incoming.decode("utf-8", errors="replace").strip()
                        if line:
                            writer.writerow([datetime.now().isoformat(timespec="milliseconds"),
                                             round(time.monotonic() - start, 4), line])
    except KeyboardInterrupt:
        print(f"Saved serial data: {folder}")
    except (serial.SerialException, OSError) as exc:
        print(f"Serial error: {exc}")


if __name__ == "__main__":
    main()
