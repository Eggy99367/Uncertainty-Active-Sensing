#include <Arduino.h>

// SETTINGS
static const char* WIFI_SSID = "CSI_Experiment";
static const char* WIFI_PASSWORD = "88888888";
static const uint32_t PC_BAUD = 921600;
static const uint32_t RADAR_BAUD = 256000;
static const int RADAR_RX_PIN = 16; // ESP32 RX2 <- LD2450 TX
static const int RADAR_TX_PIN = 17; // ESP32 TX2 -> LD2450 RX
static const uint32_t PING_INTERVAL_MS = 100;

HardwareSerial radar(2);
void setup() {
  Serial.begin(PC_BAUD);
  radar.begin(RADAR_BAUD, SERIAL_8N1, RADAR_RX_PIN, RADAR_TX_PIN);
}

void loop() {
  while (radar.available()) Serial.write(radar.read());
}
