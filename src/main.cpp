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
static uint8_t radarFrame[30];
static size_t radarIndex = 0;
static int16_t decodeSignedMagnitude(uint8_t lo, uint8_t hi) {
  uint16_t raw = (uint16_t)lo | ((uint16_t)hi << 8);
  return (raw & 0x8000) ? (int16_t)(raw & 0x7FFF) : -(int16_t)(raw & 0x7FFF);
}
static uint16_t decodeU16(uint8_t lo, uint8_t hi) {
  return (uint16_t)lo | ((uint16_t)hi << 8);
}
static void emitRadarFrame() {
  const uint32_t timestamp = millis();
  for (int target = 0; target < 3; ++target) {
    const int i = 4 + target * 8;
    const int16_t x = decodeSignedMagnitude(radarFrame[i], radarFrame[i + 1]);
    const int16_t y = decodeSignedMagnitude(radarFrame[i + 2], radarFrame[i + 3]);
    const int16_t speed = decodeSignedMagnitude(radarFrame[i + 4], radarFrame[i + 5]);
    const uint16_t resolution = decodeU16(radarFrame[i + 6], radarFrame[i + 7]);
    const bool detected = (x != 0 || y != 0 || speed != 0 || resolution != 0);
    Serial.printf("RADAR_DATA,%lu,%d,%d,%d,%d,%u,%d\n",
                  (unsigned long)timestamp, target + 1, (int)x, (int)y,
                  (int)speed, (unsigned)resolution, detected ? 1 : 0);
  }
}
static void processRadar() {
  while (radar.available()) {
    const uint8_t b = (uint8_t)radar.read();
    if (radarIndex == 0 && b != 0xAA) continue;
    radarFrame[radarIndex++] = b;
    if (radarIndex == sizeof(radarFrame)) {
      if (radarFrame[28] == 0x55 && radarFrame[29] == 0xCC) emitRadarFrame();
      radarIndex = 0;
    }
  }
}

void setup() {
  Serial.begin(PC_BAUD);
  radar.begin(RADAR_BAUD, SERIAL_8N1, RADAR_RX_PIN, RADAR_TX_PIN);
}

void loop() {
  processRadar();
}
