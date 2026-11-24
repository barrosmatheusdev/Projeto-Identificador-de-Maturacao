#pragma once
// Configuração do protótipo. Tudo que depende da montagem fica aqui.
// Valores marcados com (CALIBRAR) precisam ser ajustados na bancada.

// ===================== Modo inicial =====================
// 0 = embarcado (inferência no ESP32, Wi-Fi desligado: R04)
// 1 = remoto (envia a imagem ao servidor Flask: R06)
// O modo pode ser trocado pelo monitor serial com o comando "m".
#define MODO_INICIAL_REMOTO 0

// ===================== Pinos =====================
// Câmera usa 4-13, 15-18; PSRAM OPI usa 35-37. Confirmar no pinout da Freenove.
#define PINO_SERVO      14   // comporta (MG90S); -1 desliga o servo (teste de câmera)
#define PINO_BATENTE    -1   // servo do batente do modo remoto; -1 = sem batente

// Sensor de presença antes da câmera
#define SENSOR_ULTRASSOM 0   // 0 = infravermelho digital; 1 = ultrassônico HC-SR04
#define PINO_SENSOR     21   // saída digital do sensor IR
#define SENSOR_NIVEL_DETECTA LOW   // módulos IR comuns (FC-51/TCRT5000) vão a LOW com objeto
#define PINO_TRIG       47   // HC-SR04 (só se SENSOR_ULTRASSOM = 1)
#define PINO_ECHO       21   // HC-SR04: ECHO é 5 V, usar divisor de tensão para 3,3 V
#define LIMIAR_CM       8    // (CALIBRAR) distância abaixo da qual há fruto

// ===================== Servos =====================
#define ANG_MADURO      45   // (CALIBRAR) comporta desvia para "maduro"
#define ANG_VERDE       135  // (CALIBRAR) comporta desvia para "verde"
#define ANG_FALHA       90   // (CALIBRAR) compartimento lateral, sem resposta do servidor (R06)
#define ANG_BATENTE_FECHADO 0
#define ANG_BATENTE_ABERTO  90
#define SERVO_PULSO_MIN 500  // µs, faixa do MG90S
#define SERVO_PULSO_MAX 2400

// ===================== Tempos (ms) =====================
#define ATRASO_CAPTURA_MS   0     // (CALIBRAR) do sensor até o fruto estar no quadro da câmera
#define DESCARTAR_QUADRO    1     // 1 = descarta o quadro que já estava no buffer (pode ser antigo)
#define TEMPO_PASSAGEM_MS   800   // (CALIBRAR) batente aberto até o fruto passar
#define TEMPO_REARME_MS     300   // sensor livre por este tempo antes de aceitar outro fruto
#define LIMITE_REMOTO_MS    2000  // R06

// ===================== Inferência =====================
// Onde fica a tensor arena: 1 = PSRAM (texto do TCC), 0 = RAM interna (mais rápida, mas
// disputa memória com o Wi-Fi do modo remoto). O uso real é impresso no boot (R05).
// Medido em 2026-11-14 com a arena na PSRAM: 212.812 bytes usados, Invoke() de 204 ms.
#define ARENA_EM_PSRAM  1
#define ARENA_BYTES     (ARENA_EM_PSRAM ? 2 * 1024 * 1024 : 224 * 1024)
#define LIMIAR_MADURO   0.5f                // p(maduro) mínima; só vale se o modelo.h não trouxer MODELO_LIMIAR_MADURO
#define REPETICOES_BENCHMARK 50
