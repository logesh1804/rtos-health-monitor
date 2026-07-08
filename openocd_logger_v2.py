"""
openocd_logger_v2.py
---------------------
V2 alternative data source for the RTOS Health Predictor.

Reads the live RTOS_Health_t struct directly out of STM32 RAM via OpenOCD's
telnet interface (localhost:4444), decodes it, and appends rows to the
SAME CSV schema used by capture_uart.py / label_data.py.

This module does NOT touch UART, AI training, prediction, or the dashboard.
It is a second, independent way to fill data/real_capture.csv.

Pipeline:
    STM32 --> Health struct in RAM --> OpenOCD (localhost:4444)
          --> openocd_logger_v2.py --> data/real_capture.csv (existing schema)
          --> label_data.py / fix_and_retrain.py (unchanged)

Usage:
    python openocd_logger_v2.py --samples 100
    python openocd_logger_v2.py --address 0x2000007C --words 7 --period 1.0
    python openocd_logger_v2.py --queue-address 0x20000abc

After capturing, label exactly as with UART data:
    python label_data.py --input data/real_capture.csv --label 0
"""

import argparse
import csv
import os
import socket
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# Schema — MUST stay identical to capture_uart.py / label_data.py
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "cpu_load", "queue_level", "queue_dropped", "process_jitter",
    "process_deadline_miss", "process_exec_time", "process_stack_left",
    "sensor_deadline_miss", "comm_deadline_miss",
]

OUT_FILE = "data/real_capture.csv"

# Order the Health struct fields are laid out in RAM (matches main.c)
# RTOS_Health_t: CpuLoad, ProcessExecTime, ProcessJitter, QueueDropped,
#                ProcessDeadlineMiss, SensorDeadlineMiss, CommDeadlineMiss,
#                ProcessStackLeft
HEALTH_FIELD_ORDER = [
    "CpuLoad",
    "ProcessExecTime",
    "ProcessJitter",
    "QueueDropped",
    "ProcessDeadlineMiss",
    "SensorDeadlineMiss",
    "CommDeadlineMiss",
    "ProcessStackLeft",
    "QueueLevel",
]

DEFAULT_HEALTH_ADDRESS = 0x2000007C
DEFAULT_HEALTH_WORDS = 9  # 8 x uint32_t fields in RTOS_Health_t (V2.1: includes ProcessStackLeft)


class OpenOCDConnectionError(Exception):
    """Raised when OpenOCD cannot be reached or the connection drops."""
    pass


class OpenOCDClient:
    """
    Minimal telnet-protocol client for OpenOCD's command server
    (localhost:4444 by default). Sends raw OpenOCD commands and
    parses the text response.
    """

    PROMPT = b"> "

    def __init__(self, host="localhost", port=4444, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        try:
            self.sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            )
            self.sock.settimeout(self.timeout)
            # OpenOCD sends a banner + first prompt on connect; drain it.
            self._read_until_prompt()
        except (socket.error, OSError) as e:
            self.sock = None
            raise OpenOCDConnectionError(f"Could not connect to {self.host}:{self.port}: {e}")

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def is_connected(self):
        return self.sock is not None

    def _read_until_prompt(self):
        buf = b""
        self.sock.settimeout(self.timeout)
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise OpenOCDConnectionError("OpenOCD closed the connection")
            buf += chunk
            if self.PROMPT in buf:
                break
        return buf

    def send_command(self, command: str) -> str:
        """Send a single OpenOCD command and return its text response."""
        if not self.sock:
            raise OpenOCDConnectionError("Not connected to OpenOCD")
        try:
            self.sock.sendall((command + "\n").encode("utf-8"))
            raw = self._read_until_prompt()
        except (socket.timeout, socket.error, OSError) as e:
            self.sock = None
            raise OpenOCDConnectionError(f"Communication error during '{command}': {e}")

        text = raw.decode("utf-8", errors="ignore")
        # Strip the echoed command and trailing prompt
        text = text.replace(command, "", 1)
        text = text.replace("> ", "")
        return text.strip()

    def read_memory_words(self, address: int, count: int):
        """
        Reads `count` 32-bit words starting at `address` using OpenOCD's
        `mdw` command. Returns a list of ints in the order read.
        """
        command = f"mdw 0x{address:08X} {count}"
        response = self.send_command(command)
        return self._parse_mdw_response(response, expected_count=count)

    @staticmethod
    def _parse_mdw_response(response: str, expected_count: int):
        """
        Parses OpenOCD `mdw` output, e.g.:

            0x2000007c: 0000001e 00000180 00000032 00000000 00000003 00000001 00000002

        Returns a list of decoded decimal ints.
        """
        values = []
        for line in response.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            _, _, hex_part = line.partition(":")
            for token in hex_part.strip().split():
                try:
                    values.append(int(token, 16))
                except ValueError:
                    continue

        if len(values) < expected_count:
            raise OpenOCDConnectionError(
                f"Health structure length mismatch: expected {expected_count} "
                f"words, got {len(values)}. Raw response: {response!r}"
            )

        return values[:expected_count]


class HealthLogger:
    """
    Object-oriented OpenOCD runtime health logger.

    Connects to OpenOCD, periodically reads the RTOS_Health_t struct
    (and optionally queue_level / process_stack_left from separate
    addresses, if configured), and appends rows to the existing
    AI dataset CSV using the unmodified schema.
    """

    def __init__(
        self,
        host="localhost",
        port=4444,
        health_address=DEFAULT_HEALTH_ADDRESS,
        health_words=DEFAULT_HEALTH_WORDS,
        period=1.0,
        out_file=OUT_FILE,
        queue_address=None,
        reconnect_delay=2.0,
        max_reconnect_attempts=5,
    ):
        self.host = host
        self.port = port
        self.health_address = health_address
        self.health_words = health_words
        self.period = period
        self.out_file = out_file
        self.queue_address = queue_address
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self.client = OpenOCDClient(host=host, port=port)
        self.samples_collected = 0
        self.errors_count = 0

    # ── Connection management ──────────────────────────────────────────────
    def connect(self):
        print(f"[openocd_logger_v2] Connecting to OpenOCD at {self.host}:{self.port} ...")
        self.client.connect()
        print("[openocd_logger_v2] Connected.")

    def ensure_connected(self):
        """Reconnects automatically if the OpenOCD link has dropped."""
        if self.client.is_connected():
            return True

        for attempt in range(1, self.max_reconnect_attempts + 1):
            print(f"[openocd_logger_v2] Reconnect attempt {attempt}/{self.max_reconnect_attempts} ...")
            try:
                self.client.connect()
                print("[openocd_logger_v2] Reconnected.")
                return True
            except OpenOCDConnectionError as e:
                print(f"[openocd_logger_v2] Reconnect failed: {e}")
                time.sleep(self.reconnect_delay)

        return False

    # ── Health struct decoding ─────────────────────────────────────────────
    def read_health_struct(self) -> dict:
        """
        Reads the RTOS_Health_t struct from RAM and returns a dict
        keyed by the C struct field names (CpuLoad, ProcessExecTime, ...).
        """
        words = self.client.read_memory_words(self.health_address, self.health_words)

        if len(words) != len(HEALTH_FIELD_ORDER):
            raise OpenOCDConnectionError(
                f"Decoded word count ({len(words)}) does not match "
                f"expected Health struct fields ({len(HEALTH_FIELD_ORDER)})"
            )

        return dict(zip(HEALTH_FIELD_ORDER, words))

    def read_optional_word(self, address):
        """Reads a single extra uint32_t (queue_level / process_stack_left)."""
        if address is None:
            return 0
        try:
            words = self.client.read_memory_words(address, 1)
            return words[0]
        except OpenOCDConnectionError as e:
            print(f"[openocd_logger_v2] Warning: optional read failed ({e}); using 0")
            return 0

    def build_feature_row(self) -> list:
        """
        Builds one CSV row matching FEATURE_COLS exactly:

            cpu_load, queue_level, queue_dropped, process_jitter,
            process_deadline_miss, process_exec_time, process_stack_left,
            sensor_deadline_miss, comm_deadline_miss
        """
        health = self.read_health_struct()

        queue_level = self.read_optional_word(self.queue_address)

        row = {
            "cpu_load": health["CpuLoad"],
            "queue_level": health["QueueLevel"],  # now from struct directly
            "queue_dropped": health["QueueDropped"],
            "process_jitter": health["ProcessJitter"],
            "process_deadline_miss": health["ProcessDeadlineMiss"],
            "process_exec_time": health["ProcessExecTime"],
            "process_stack_left": health["ProcessStackLeft"],
            "queue_level_live": health["QueueLevel"],
            "sensor_deadline_miss": health["SensorDeadlineMiss"],
            "comm_deadline_miss": health["CommDeadlineMiss"],
        }

        return [row[col] for col in FEATURE_COLS]

    # ── CSV output ──────────────────────────────────────────────────────────
    def _open_csv_writer(self, csvfile):
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            # New file — write header, matching capture_uart.py format
            writer.writerow(FEATURE_COLS + ["health_label"])
        return writer

    # ── Main capture loop ──────────────────────────────────────────────────
    def run(self, n_samples: int):
        os.makedirs(os.path.dirname(self.out_file) or ".", exist_ok=True)

        print(f"\n[openocd_logger_v2] Health struct address : 0x{self.health_address:08X}")
        print(f"[openocd_logger_v2] Health struct words   : {self.health_words}")
        print(f"[openocd_logger_v2] Sampling period        : {self.period}s")
        print(f"[openocd_logger_v2] Output file             : {self.out_file}")
        if self.queue_address:
            print(f"[openocd_logger_v2] queue_level address     : 0x{self.queue_address:08X}")
        else:
            print(f"[openocd_logger_v2] queue_level address     : not configured (defaults to 0)")
        print(f"[openocd_logger_v2] process_stack_left          : read from Health struct (live)")
        print(f"[openocd_logger_v2] Collecting {n_samples} samples. Press Ctrl-C to stop early.\n")

        self.connect()

        with open(self.out_file, "a", newline="") as csvfile:
            writer = self._open_csv_writer(csvfile)

            while self.samples_collected < n_samples:
                if not self.ensure_connected():
                    print("[openocd_logger_v2] ERROR: Unable to reconnect to OpenOCD. Stopping.")
                    break

                try:
                    row = self.build_feature_row()
                except OpenOCDConnectionError as e:
                    self.errors_count += 1
                    print(f"[openocd_logger_v2] Read error ({e}); will retry/reconnect.")
                    self.client.close()
                    time.sleep(self.reconnect_delay)
                    continue

                # Leave health_label blank — filled in later via label_data.py
                writer.writerow(row + [""])
                csvfile.flush()
                self.samples_collected += 1

                print(
                    f"  [{self.samples_collected:>3}/{n_samples}]  "
                    f"cpu={row[0]:.0f}%  "
                    f"jitter={row[3]:.0f}ms  "
                    f"dl_miss={row[4]:.0f}  "
                    f"stack={row[6]:.0f}"
                )

                time.sleep(self.period)

        self.client.close()

        print(f"\n[openocd_logger_v2] Done. {self.samples_collected} rows appended to {self.out_file}")
        if self.errors_count:
            print(f"[openocd_logger_v2] Encountered {self.errors_count} read error(s) during capture.")
        print("\n── Next steps ──────────────────────────────────────────────")
        print("1. Open data/real_capture.csv and review the new rows")
        print("2. Label them, e.g.:")
        print("     python label_data.py --input data/real_capture.csv --label 0")
        print("3. Retrain as usual:")
        print("     python fix_and_retrain.py")
        print("────────────────────────────────────────────────────────────\n")


def parse_address(value: str) -> int:
    """Parses a hex (0x...) or decimal address string into an int."""
    return int(value, 16) if value.lower().startswith("0x") else int(value)


def main():
    parser = argparse.ArgumentParser(
        description="V2 OpenOCD runtime health logger — alternative data source to capture_uart.py"
    )
    parser.add_argument("--host", default="localhost", help="OpenOCD telnet host (default: localhost)")
    parser.add_argument("--port", type=int, default=4444, help="OpenOCD telnet port (default: 4444)")
    parser.add_argument(
        "--address", default=f"0x{DEFAULT_HEALTH_ADDRESS:08X}",
        help="Health struct base address (default: 0x2000007C)"
    )
    parser.add_argument(
        "--words", type=int, default=DEFAULT_HEALTH_WORDS,
        help="Number of 32-bit words in the Health struct (default: 7)"
    )
    parser.add_argument(
        "--queue-address", default=None,
        help="Optional address of uxQueueMessagesWaiting for queue_level (default: none -> 0)"
    )
    parser.add_argument("--period", type=float, default=1.0, help="Sampling period in seconds (default: 1.0)")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to collect (default: 100)")
    parser.add_argument("--out", default=OUT_FILE, help=f"Output CSV path (default: {OUT_FILE})")

    args = parser.parse_args()

    logger = HealthLogger(
        host=args.host,
        port=args.port,
        health_address=parse_address(args.address),
        health_words=args.words,
        period=args.period,
        out_file=args.out,
        queue_address=parse_address(args.queue_address) if args.queue_address else None,
    )

    try:
        logger.run(args.samples)
    except KeyboardInterrupt:
        print("\n[openocd_logger_v2] Stopped by user.")
        logger.client.close()
        sys.exit(0)


if __name__ == "__main__":
    main()