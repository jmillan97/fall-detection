#include <WiFi.h>
#include <WiFiUdp.h>

// ── WiFi Config ───────────────────────────────────────────
const char* SSID      = "Router? I Hardly Know Her!";
const char* PASSWORD  = "bubbl3z!";
const char* LAPTOP_IP = "192.168.4.62";
const int   UDP_PORT  = 5005;

// ── Hardware ──────────────────────────────────────────────
#define UART_RX   16
#define UART_TX   17

// ── Objects ───────────────────────────────────────────────
WiFiUDP udp;
HardwareSerial STM32Serial(1);

void setup() {
    Serial.begin(115200);

    STM32Serial.begin(115200, SERIAL_8N1, UART_RX, UART_TX);

    Serial.print("[ESP32] Connecting to: ");
    Serial.println(SSID);

    WiFi.begin(SSID, PASSWORD);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.print("[ESP32] Connected. IP: ");
    Serial.println(WiFi.localIP());

    udp.begin(UDP_PORT);
    Serial.println("[ESP32] Ready.");
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        WiFi.reconnect();
        while (WiFi.status() != WL_CONNECTED) {
            delay(500);
        }
    }

    if (STM32Serial.available()) {
        String line = STM32Serial.readStringUntil('\n');
        line.trim();
        if (line.length() == 0) return;

        udp.beginPacket(LAPTOP_IP, UDP_PORT);
        udp.print(line);
        udp.print("\n");
        udp.endPacket();

        Serial.println(line);
    }
}