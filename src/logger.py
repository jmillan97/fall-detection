import serial
import csv
import time
import os
import argparse
from datetime import datetime

def get_output_path(label):
    #create path for data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/raw", exist_ok=True)
    return f"data/raw/{timestamp}_{label}.csv"

def record(port, baud, label, duration=None):
    output_path = get_output_path(label)
    print(f"[LOGGER] Opening port {port} at {baud} baud")
    print(f"[LOGGER] Writing in {output_path}")
    print(f"[LOGGER] Ctrl+C to stop\n")

    try:
        #opens serial port and cleans garbage
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)
        ser.reset_input_buffer()

    except serial.SerialException as e: 
        print(f"[ERROR] Could not opent port: {e}")
        print(" On WSL: check >> ls /dev/tty*")
        return
    sample_count = 0
    start_time = time.time()
    with open(output_path, 'w', newline='') as csvfile:
        #new csv
        writer = csv.writer(csvfile)
        writer.writerow([
            "timestamp.ms", "waist_x", "waist_y", "waist_z", "thigh_x", "thigh_y", "thigh_z", "event_flag"
        ])
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    print(f"\n[LOGGER] DUration time limit reached ({duration}s)")
                    break
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                if not line:
                    continue
                if line.startswith('#'):
                    print(f"[stm32] {line}")
                    continue
                values = line.split(',')
                if len(values) != 8:
                    print(f"[WARN] Malformed line ({len(values)} cols): {line}")
                    continue
                writer.writerow(values)
                sample_count+=1
                if sample_count % 50 == 0:
                    elapsed = time.time() - start_time
                    hz = sample_count/elapsed
                    print(f"[LOGGER] {sample_count} sampels | {elapsed:.1f}s | {hz:.1f} Hz", end='\r')
        except KeyboardInterrupt:
            print(f"\n[LOGGER] Stopped")
        finally:
            ser.close()
            elapsed = time.time() - start_time
            print(f"[LOGGER] Session ended")
            print(f"\t Samples: {sample_count}")
            print(f"\t Duration: {elapsed:.1f}")
            print(f"\t Saved to: {output_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STM32 Fall Detection Data Logger")
    parser.add_argument('--port', default='/dev/ttyACM0', help='Serial port')
    parser.add_argument('--baud', default=115200, type=int, help='Baud rate')
    parser.add_argument('--label', default='session', help='Label for filename')
    parser.add_argument('--duration', default=None, type=float, help='Recording duration')
    args = parser.parse_args()
    record(args.port, args.baud, args.label, args.duration)


