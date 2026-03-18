import socket 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((('0.0.0.0', 5005)))

sock.settimeout(5)

print("Listening for UDP on port 5005...")

try:
    while True:
        data, addr = sock.recvfrom(256)
        print(f"Received from {addr}: {data.decode()}")
except KeyboardInterrupt:
    sock.close()
