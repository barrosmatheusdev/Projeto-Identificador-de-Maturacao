#include "remoto.h"
#include "config.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "img_converters.h"

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Crie o secrets.h a partir do secrets_exemplo.h (Wi-Fi e URL do servidor)"
#endif

static inline uint32_t agora_us() { return (uint32_t)esp_timer_get_time(); }

bool wifi_conectar(uint32_t limite_ms) {
  if (WiFi.status() == WL_CONNECTED) return true;
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);   // economia de energia aumenta a latência
  WiFi.begin(WIFI_SSID, WIFI_SENHA);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < limite_ms) delay(100);
  if (WiFi.status() != WL_CONNECTED) return false;
  Serial.printf("INFO,wifi,%s,rssi=%d\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
  return true;
}

void wifi_desligar() {
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
}

bool wifi_ok() { return WiFi.status() == WL_CONNECTED; }

static float campo(const String& json, const char* chave, float padrao = NAN) {
  String alvo = String("\"") + chave + "\":";
  int i = json.indexOf(alvo);
  if (i < 0) return padrao;
  return json.substring(i + alvo.length()).toFloat();
}

bool remoto_classificar(camera_fb_t* fb, int fruto, Resultado& r, TemposRemoto& t, String& erro) {
  uint32_t t0 = agora_us();
  uint8_t* jpg = nullptr;
  size_t n = 0;
  if (!frame2jpg(fb, 80, &jpg, &n)) {
    erro = "falha ao gerar JPEG";
    return false;
  }
  uint32_t t1 = agora_us();
  t.t_jpeg_us = t1 - t0;

  bool ok = false;
  if (!wifi_ok()) {
    erro = "sem Wi-Fi";
  } else {
    HTTPClient http;
    String url = String(SERVIDOR_URL) + "/classificar?fruto=" + fruto;
    http.begin(url);
    http.setConnectTimeout(LIMITE_REMOTO_MS);
    http.setTimeout(LIMITE_REMOTO_MS);
    http.addHeader("Content-Type", "image/jpeg");
    int cod = http.POST(jpg, n);
    if (cod == 200) {
      String resp = http.getString();
      int idx = (int)campo(resp, "indice", -1);
      if (idx == 0 || idx == 1) {
        r.classe = idx;
        r.conf = campo(resp, "confianca", 0);
        r.probs[idx] = r.conf;
        r.probs[1 - idx] = 1 - r.conf;
        t.t_servidor_ms = campo(resp, "t_servidor_ms", 0);
        ok = true;
      } else {
        erro = "resposta sem classe";
      }
    } else {
      erro = String("HTTP ") + cod + " " + HTTPClient::errorToString(cod);
    }
    http.end();
  }
  free(jpg);
  t.t_rede_us = agora_us() - t1;
  if (ok && t.t_rede_us > (uint32_t)LIMITE_REMOTO_MS * 1000) {   // R06: chegou, mas tarde demais
    erro = "resposta depois do limite";
    ok = false;
  }
  return ok;
}

bool remoto_bruto(camera_fb_t* fb, String& resposta) {
  if (!wifi_ok()) {
    resposta = "sem Wi-Fi";
    return false;
  }
  HTTPClient http;
  String url = String(SERVIDOR_URL) + "/classificar?w=" + fb->width + "&h=" + fb->height;
  http.begin(url);
  http.setTimeout(10000);
  http.addHeader("Content-Type", "image/x-rgb565");
  int cod = http.POST(fb->buf, fb->len);
  resposta = cod > 0 ? http.getString() : HTTPClient::errorToString(cod);
  http.end();
  return cod == 200;
}
