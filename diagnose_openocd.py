"""
diagnose_openocd.py
--------------------
Standalone diagnostic — connects to OpenOCD and prints every step,
so we can see exactly where things hang or fail, independent of
server.py or any background threads.

Usage:
    python diagnose_openocd.py
"""

import socket
import sys

HOST = "localhost"
PORT = 4444
ADDRESS = 0x2000007C
WORDS = 8


def main():
    print(f"[1] Connecting to {HOST}:{PORT} ...")
    try:
        sock = socket.create_connection((HOST, PORT), timeout=5.0)
    except Exception as e:
        print(f"[FAIL] Could not connect: {e}")
        sys.exit(1)
    print("[2] TCP connection established.")

    sock.settimeout(5.0)

    print("[3] Draining initial banner/prompt ...")
    try:
        buf = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                print("[FAIL] Connection closed while draining banner.")
                sys.exit(1)
            buf += chunk
            print(f"    received {len(chunk)} bytes: {chunk!r}")
            if b"> " in buf:
                break
    except socket.timeout:
        print("[FAIL] Timed out waiting for initial prompt. Raw buffer so far:")
        print(f"    {buf!r}")
        sys.exit(1)
    print(f"[4] Banner drained OK. Full banner was: {buf!r}")

    command = f"mdw 0x{ADDRESS:08X} {WORDS}"
    print(f"[5] Sending command: {command!r}")
    try:
        sock.sendall((command + "\n").encode("utf-8"))
    except Exception as e:
        print(f"[FAIL] Send failed: {e}")
        sys.exit(1)

    print("[6] Waiting for response ...")
    try:
        buf = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                print("[FAIL] Connection closed while waiting for response.")
                sys.exit(1)
            buf += chunk
            print(f"    received {len(chunk)} bytes: {chunk!r}")
            if b"> " in buf:
                break
    except socket.timeout:
        print("[FAIL] Timed out waiting for mdw response. Raw buffer so far:")
        print(f"    {buf!r}")
        sys.exit(1)

    print(f"[7] SUCCESS. Full response: {buf!r}")
    sock.close()
    print("[8] Done.")


if __name__ == "__main__":
    main()