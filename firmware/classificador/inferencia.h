#pragma once
#include <Arduino.h>
#include "esp_camera.h"

struct Resultado {
  int classe = -1;            // 0 = verde, 1 = maduro, -1 = sem decisão
  float conf = 0;             // probabilidade da classe escolhida
  float probs[2] = {0, 0};
  int8_t bruto[2] = {0, 0};   // saída int8 do modelo (para comparar com o PC)
  uint32_t t_preproc_us = 0;
  uint32_t t_inferencia_us = 0;
};

bool inferencia_iniciar(String& erro);
void inferencia_info();                                  // imprime tamanho, arena usada, quantização
bool inferencia_classificar(const camera_fb_t* fb, Resultado& r);
long inferencia_soma_entrada();                          // soma dos int8 da entrada da última classificação
void inferencia_benchmark(const camera_fb_t* fb, int n); // só o Invoke(), n vezes
void media_cor_centro(const camera_fb_t* fb);            // diagnóstico de ordem dos canais
