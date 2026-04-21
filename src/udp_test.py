import socket
import time

ESP32_IP   = "192.168.4.66"   # ← update with ESP32 IP from Serial Monitor
ESP32_PORT = 5005
LOCAL_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', LOCAL_PORT))
sock.settimeout(5)

print(f"[udp_test] Sending REGISTER to ESP32 at {ESP32_IP}:{ESP32_PORT}")
sock.sendto(b"REGISTER", (ESP32_IP, ESP32_PORT))

try:
    data, addr = sock.recvfrom(256)
    print(f"[udp_test] Response from {addr}: {data.decode().strip()}")
except socket.timeout:
    print("[udp_test] No response from ESP32 — check IP and network")
    sock.close()
    exit()

print("[udp_test] Listening for data...")

try:
    while True:
        try:
            data, addr = sock.recvfrom(256)
            print(f"Received from {addr}: {data.decode().strip()}")
        except socket.timeout:
            print("waiting...", end='\r')
except KeyboardInterrupt:
    sock.close()
    print("\n[udp_test] Done")