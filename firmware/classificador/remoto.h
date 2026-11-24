#pragma once
#include <Arduino.h>
#include "esp_camera.h"
#include "inferencia.h"

struct TemposRemoto {
  uint32_t t_jpeg_us = 0;       // RGB565 -> JPEG (qualidade 80, igual ao /capture da coleta)
  uint32_t t_rede_us = 0;       // POST até a resposta completa
  float t_servidor_ms = 0;      // tempo informado pelo servidor
};

bool wifi_conectar(uint32_t limite_ms);
void wifi_desligar();
bool wifi_ok();

// Modo remoto: envia o quadro como JPEG e lê a classe. false = sem resposta válida dentro do limite.
bool remoto_classificar(camera_fb_t* fb, int fruto, Resultado& r, TemposRemoto& t, String& erro);

// Envia o RGB565 cru, para o servidor fazer exatamente o mesmo pré-processamento do ESP32
// (usado pelo comando "x" para comparar o int8 do chip com o int8 no PC).
bool remoto_bruto(camera_fb_t* fb, String& resposta);
