#include <Arduino.h>
#include <WiFi.h>
#include <ESP32Ping.h>
#include <esp_wifi.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

// SETTINGS
static const char* WIFI_SSID = "CSI_Experiment";
static const char* WIFI_PASSWORD = "88888888";
static const uint32_t PC_BAUD = 921600;
static const uint32_t RADAR_BAUD = 256000;
static const int RADAR_RX_PIN = 16; // ESP32 RX2 <- LD2450 TX
static const int RADAR_TX_PIN = 17; // ESP32 TX2 -> LD2450 RX
static const uint32_t PING_INTERVAL_MS = 100;

HardwareSerial radar(2);
struct CsiPacket {
  uint32_t timestamp;
  int8_t rssi;
  uint16_t len;
  uint8_t firstInvalid;
  int8_t iq[384];
};
static QueueHandle_t csiQueue;
static volatile uint32_t csiDropped = 0;

static void csiCallback(void*, wifi_csi_info_t* info) {
  if (!info || !info->buf || !csiQueue) return;
  CsiPacket packet{};
  packet.timestamp = millis();
  packet.rssi = info->rx_ctrl.rssi;
  packet.len = (info->len <= sizeof(packet.iq)) ? info->len : sizeof(packet.iq);
  packet.firstInvalid = info->first_word_invalid ? 1 : 0;
  memcpy(packet.iq, info->buf, packet.len);
  if (xQueueSend(csiQueue, &packet, 0) != pdTRUE) csiDropped++;
}

static void emitCsi(const CsiPacket& p) {
  Serial.printf("CSI_DATA,%lu,%d,%u,%u,\"", (unsigned long)p.timestamp,
                (int)p.rssi, (unsigned)p.len, (unsigned)p.firstInvalid);
  for (uint16_t i = 0; i < p.len; ++i) {
    if (i) Serial.write(' ');
    Serial.print((int)p.iq[i]);
  }
  Serial.println('"');
}

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
  csiQueue = xQueueCreate(24, sizeof(CsiPacket));
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) delay(200);
  if (WiFi.status() == WL_CONNECTED) {
    wifi_csi_config_t config{};
    config.lltf_en = true;
    config.htltf_en = true;
    config.stbc_htltf2_en = true;
    config.ltf_merge_en = true;
    config.channel_filter_en = false;
    config.manu_scale = false;
    config.shift = 0;
    esp_err_t a = esp_wifi_set_csi_rx_cb(&csiCallback, nullptr);
    esp_err_t b = esp_wifi_set_csi_config(&config);
    esp_err_t c = esp_wifi_set_csi(true);
    Serial.printf("STATUS,WIFI_CONNECTED,%s\n", WiFi.localIP().toString().c_str());
    Serial.printf("STATUS,CSI_INIT,%d,%d,%d\n", (int)a, (int)b, (int)c);
  } else {
    Serial.println("STATUS,WIFI_FAILED");
  }
  Serial.println("STATUS,RADAR_UART_READY");
}

void loop() {
  processRadar();
  CsiPacket packet;
  if (xQueueReceive(csiQueue, &packet, 0) == pdTRUE) emitCsi(packet);
  processRadar();
  static uint32_t lastPing = 0;
  if (WiFi.status() == WL_CONNECTED && millis() - lastPing >= PING_INTERVAL_MS) {
    lastPing = millis();
    Ping.ping(WiFi.gatewayIP(), 1);
  }
  static uint32_t lastStatus = 0;
  if (millis() - lastStatus >= 5000) {
    lastStatus = millis();
    Serial.printf("STATUS,CSI_DROPPED,%lu\n", (unsigned long)csiDropped);
  }
}
