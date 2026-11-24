// Classificador de maturação de tomates (verde / maduro) — ESP32-S3 N16R8 + OV5640 (Freenove)
//
// Placa: ESP32S3 Dev Module | Flash 16 MB | Partition Scheme 16 MB | PSRAM: OPI PSRAM
// Núcleo: esp32 da Espressif 3.x (traz o esp-tflite-micro com ESP-NN). Biblioteca: ESP32Servo.
// Antes de compilar: copiar modelo.h e modelo.cpp gerados pelo notebook para esta pasta
// e criar o secrets.h a partir do secrets_exemplo.h.
//
// Cada fruto gera uma linha "LOG,..." no monitor serial (R07). Comandos: envie "h".

#include <ESP32Servo.h>
#include "config.h"
#include "camera.h"
#include "inferencia.h"
#include "remoto.h"
#include "img_converters.h"

enum Modo { EMBARCADO = 0, REMOTO = 1 };
static Modo modo = MODO_INICIAL_REMOTO ? REMOTO : EMBARCADO;
static Servo comporta;
#if PINO_BATENTE >= 0
static Servo batente;
#endif
static int fruto = 1;

// A tarefa do loop() tem 8 KB de pilha por padrão; o codificador JPEG (frame2jpg) mais o
// HTTPClient do modo remoto estouram isso e a placa reinicia. 32 KB dão folga.
SET_LOOP_TASK_STACK_SIZE(32 * 1024);

static inline uint32_t agora_us() { return (uint32_t)esp_timer_get_time(); }

// ---------------------------------------------------------------- sensor
static bool sensor_detecta() {
#if SENSOR_ULTRASSOM
  digitalWrite(PINO_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PINO_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PINO_TRIG, LOW);
  unsigned long us = pulseIn(PINO_ECHO, HIGH, 25000);
  return us > 0 && us / 58.0 < LIMIAR_CM;
#else
  return digitalRead(PINO_SENSOR) == SENSOR_NIVEL_DETECTA;
#endif
}

static void esperar_sensor_livre() {
  uint32_t livre_desde = millis();
  while (millis() - livre_desde < TEMPO_REARME_MS) {
    if (sensor_detecta()) livre_desde = millis();
    delay(5);
  }
}

// ---------------------------------------------------------------- modos
static void entrar_modo(Modo m) {
  modo = m;
  if (modo == EMBARCADO) {
    wifi_desligar();   // R04: ciclo completo sem rede
    Serial.println("INFO,modo,embarcado (Wi-Fi desligado)");
  } else {
    Serial.println("INFO,modo,remoto (conectando ao Wi-Fi...)");
    if (!wifi_conectar(15000)) Serial.println("ERRO,wifi,não conectou; frutos irão para o compartimento lateral");
  }
}

// ---------------------------------------------------------------- ciclo de um fruto
static void ciclo(bool manual) {
  uint32_t t_sensor = agora_us();
  if (!manual && ATRASO_CAPTURA_MS > 0) delay(ATRASO_CAPTURA_MS);

  camera_fb_t* fb = camera_capturar();
  uint32_t t_captura = agora_us();
  if (!fb) {
    Serial.println("ERRO,camera,quadro nulo");
    comporta.write(ANG_FALHA);
    return;
  }

  Resultado r;
  TemposRemoto tr;
  String status = "ok";
  bool ok;
  if (modo == EMBARCADO) {
    ok = inferencia_classificar(fb, r);
    if (!ok) status = "erro_inferencia";
  } else {
    String erro;
    ok = remoto_classificar(fb, fruto, r, tr, erro);
    if (!ok) status = "falha_remoto:" + erro;
  }
  esp_camera_fb_return(fb);

  int angulo = !ok ? ANG_FALHA : (r.classe == 1 ? ANG_MADURO : ANG_VERDE);
  comporta.write(angulo);
  uint32_t t_comando = agora_us();

#if PINO_BATENTE >= 0
  if (modo == REMOTO) {   // fruto ficou parado no batente até a decisão
    delay(150);           // dá tempo de a comporta chegar antes de liberar
    batente.write(ANG_BATENTE_ABERTO);
    delay(TEMPO_PASSAGEM_MS);
    batente.write(ANG_BATENTE_FECHADO);
  }
#endif

  const char* nome = !ok ? "lateral" : (r.classe == 1 ? "maduro" : "verde");
  // 8 tempos em %.2f: sensor→captura, preproc, inferência, jpeg, rede, servidor, captura→comporta, total
  Serial.printf("LOG,%d,%s,%s,%.4f,%.4f,%.4f,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s\n",
                fruto, modo == EMBARCADO ? "embarcado" : "remoto", nome, r.conf, r.probs[0], r.probs[1],
                r.bruto[0], r.bruto[1],
                (t_captura - t_sensor) / 1000.0, r.t_preproc_us / 1000.0, r.t_inferencia_us / 1000.0,
                tr.t_jpeg_us / 1000.0, tr.t_rede_us / 1000.0, tr.t_servidor_ms,
                (t_comando - t_captura) / 1000.0, (t_comando - t_sensor) / 1000.0, status.c_str());
  fruto++;
}

// ---------------------------------------------------------------- comandos seriais
static void ajuda() {
  Serial.println(
    "Comandos (enviar com Enter):\n"
    "  t        ciclo manual (sem sensor)\n"
    "  m        alterna modo embarcado / remoto\n"
    "  f N      próximo fruto passa a ser N\n"
    "  b        benchmark: Invoke() repetido sobre um quadro novo\n"
    "  v        prévia: manda o quadro em JPEG pela serial (IMG,<bytes> + dados)\n"
    "  c        média de R, G, B no centro da imagem (conferir ordem das cores)\n"
    "  x        compara a saída int8 do ESP32 com a do servidor no mesmo quadro\n"
    "  s        testa a comporta (maduro, verde, lateral)\n"
    "  i        informações de memória e do modelo\n"
    "  h        esta ajuda");
}

static void comando(String lin) {
  lin.trim();
  if (lin.isEmpty()) return;
  char c = lin[0];
  if (c == 't') {
    ciclo(true);
  } else if (c == 'm') {
    entrar_modo(modo == EMBARCADO ? REMOTO : EMBARCADO);
  } else if (c == 'f') {
    fruto = lin.substring(1).toInt();
    Serial.printf("INFO,proximo_fruto,%d\n", fruto);
  } else if (c == 'b' || c == 'c' || c == 'x') {
    // no "x" o Wi-Fi liga ANTES da captura, para o quadro não ficar parado esperando a conexão
    bool desligado = (c == 'x') && !wifi_ok();
    if (desligado && !wifi_conectar(15000)) { Serial.println("ERRO,wifi,não conectou"); return; }
    camera_fb_t* fb = camera_capturar();
    if (!fb) { Serial.println("ERRO,camera,quadro nulo"); return; }
    if (c == 'b') {
      inferencia_benchmark(fb, REPETICOES_BENCHMARK);
    } else if (c == 'c') {
      media_cor_centro(fb);
    } else {
      uint32_t soma_fb_antes = 0;
      for (size_t i = 0; i < fb->len; i++) soma_fb_antes += fb->buf[i];
      Resultado r;
      inferencia_classificar(fb, r);
      Serial.printf("COMPARA,esp32,int8=[%d,%d],probs=[%.4f,%.4f],soma_entrada=%ld\n", r.bruto[0], r.bruto[1],
                    r.probs[0], r.probs[1], inferencia_soma_entrada());
      uint32_t soma_fb_depois = 0;
      for (size_t i = 0; i < fb->len; i++) soma_fb_depois += fb->buf[i];
      Serial.printf("COMPARA,quadro,soma_antes=%u,soma_envio=%u,%s\n", soma_fb_antes, soma_fb_depois,
                    soma_fb_antes == soma_fb_depois ? "igual" : "MUDOU");
      String resp;
      remoto_bruto(fb, resp);
      Serial.printf("COMPARA,servidor,%s\n", resp.c_str());
    }
    esp_camera_fb_return(fb);
    if (desligado && modo == EMBARCADO) wifi_desligar();
  } else if (c == 'v') {
    // Prévia para posicionar o fruto: VGA em JPEG (qualidade baixa, ~1 s a 115200 baud).
    camera_fb_t* fb = camera_capturar();
    if (!fb) { Serial.println("ERRO,camera,quadro nulo"); return; }
    uint8_t* jpg = nullptr;
    size_t n = 0;
    bool ok = frame2jpg(fb, 30, &jpg, &n);
    esp_camera_fb_return(fb);
    if (!ok) { Serial.println("ERRO,jpeg,falha na conversão"); return; }
    Serial.printf("IMG,%u\n", (unsigned)n);
    Serial.write(jpg, n);
    Serial.flush();
    free(jpg);
  } else if (c == 's') {
    for (int a : {ANG_MADURO, ANG_VERDE, ANG_FALHA}) {
      Serial.printf("INFO,servo,%d\n", a);
      comporta.write(a);
      delay(1000);
    }
  } else if (c == 'i') {
    inferencia_info();
    Serial.printf("INFO,pilha_loop_livre_min,%u\n", (unsigned)uxTaskGetStackHighWaterMark(NULL));
    Serial.printf("INFO,modo,%s\n", modo == EMBARCADO ? "embarcado" : "remoto");
  } else {
    ajuda();
  }
}

// ---------------------------------------------------------------- setup / loop
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== Classificador de tomates ===");
  if (!psramFound()) Serial.println("ERRO,psram,não encontrada (Tools > PSRAM > OPI PSRAM)");

  // Câmera primeiro, com a mesma configuração do Sketch_03 (XCLK no canal 0 / timer 0 do LEDC).
  if (!camera_iniciar()) {
    Serial.println("ERRO,camera,falha ao iniciar; parado");
    while (true) delay(1000);
  }

  // Servos só no timer 1 do LEDC (canais 2 e 3), longe do clock da câmera.
#if PINO_SERVO >= 0
  ESP32PWM::allocateTimer(1);
  comporta.setPeriodHertz(50);
  comporta.attach(PINO_SERVO, SERVO_PULSO_MIN, SERVO_PULSO_MAX);
  comporta.write(ANG_VERDE);
#else
  Serial.println("INFO,servo,desligado (PINO_SERVO = -1)");   // write() sem attach não faz nada
#endif
#if PINO_BATENTE >= 0
  batente.setPeriodHertz(50);
  batente.attach(PINO_BATENTE, SERVO_PULSO_MIN, SERVO_PULSO_MAX);
  batente.write(ANG_BATENTE_FECHADO);
#endif

#if SENSOR_ULTRASSOM
  pinMode(PINO_TRIG, OUTPUT);
  pinMode(PINO_ECHO, INPUT);
#else
  pinMode(PINO_SENSOR, INPUT_PULLUP);
#endif

  String erro;
  if (!inferencia_iniciar(erro)) {
    Serial.printf("ERRO,modelo,%s; parado\n", erro.c_str());   // R05
    while (true) delay(1000);
  }
  inferencia_info();
  entrar_modo(modo);
  Serial.println("CSV,fruto,modo,classe,confianca,p_verde,p_maduro,int8_verde,int8_maduro,"
                 "t_sensor_captura_ms,t_preproc_ms,t_inferencia_ms,t_jpeg_ms,t_rede_ms,t_servidor_ms,"
                 "t_captura_comporta_ms,t_total_ms,status");
  ajuda();
}

void loop() {
  if (Serial.available()) comando(Serial.readStringUntil('\n'));
  if (sensor_detecta()) {
    ciclo(false);
    esperar_sensor_livre();
  }
  delay(1);
}
