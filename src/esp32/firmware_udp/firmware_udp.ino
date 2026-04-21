#include <WiFi.h>
#include <WiFiUdp.h>

// ── WiFi Config ───────────────────────────────────────────
const char* SSID      = "Router? I Hardly Know Her!";
const char* PASSWORD  = "bubbl3z!";
const char* LAPTOP_IP = "192.168.4.62";   // ← update with current laptop IP
const int   UDP_PORT = 5005;

#define UART_RX 16
#define UART_TX 17

WiFiUDP udp;
HardwareSerial STM32Serial(1);

IPAddress laptopIP;
uint16_t laptopPort = 0;
bool registered = false;

void setup() {
    Serial.begin(115200);
    STM32Serial.begin(115200, SERIAL_8N1, UART_RX, UART_TX);

    WiFi.begin(SSID, PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());

    udp.begin(UDP_PORT);
    Serial.println("[ESP32] Waiting for registration from laptop...");
}

void loop() {
    // Check for registration packet from laptop
    int packetSize = udp.parsePacket();
    if (packetSize) {
        char buf[256];
        udp.read(buf, packetSize);
        buf[packetSize] = 0;

        if (String(buf).startsWith("REGISTER")) {
            laptopIP   = udp.remoteIP();
            laptopPort = udp.remotePort();
            registered = true;
            Serial.print("[ESP32] Registered laptop: ");
            Serial.print(laptopIP);
            Serial.print(":");
            Serial.println(laptopPort);

            // Confirm registration
            udp.beginPacket(laptopIP, laptopPort);
            udp.print("REGISTERED\n");
            udp.endPacket();
        }
    }

    // Forward STM32 data to registered laptop
    if (registered && STM32Serial.available()) {
        String line = STM32Serial.readStringUntil('\n');
        line.trim();
        if (line.length() == 0) return;

        udp.beginPacket(laptopIP, laptopPort);
        udp.print(line);
        udp.print("\n");
        udp.endPacket();

        Serial.println(line);
    }
}