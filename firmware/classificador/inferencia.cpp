// Inferência embarcada com o esp-tflite-micro que vem no núcleo ESP32 3.x do Arduino
// (já compilado com os kernels ESP-NN otimizados para o ESP32-S3).
#include "inferencia.h"
#include "config.h"
#include "modelo.h"
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Limiar de decisão: o do modelo.h (escolhido na validação pelo notebook) tem prioridade.
#ifdef MODELO_LIMIAR_MADURO
static const float LIMIAR = MODELO_LIMIAR_MADURO;
#else
static const float LIMIAR = LIMIAR_MADURO;
#endif

static tflite::MicroMutableOpResolver<16> resolver;
static tflite::MicroInterpreter* interp = nullptr;
static uint8_t* arena = nullptr;
static TfLiteTensor* entrada = nullptr;
static TfLiteTensor* saida = nullptr;
static int lado = 0;
static long soma_entrada;
static long somar_entrada();

static inline uint32_t agora_us() { return (uint32_t)esp_timer_get_time(); }

bool inferencia_iniciar(String& erro) {
  const tflite::Model* modelo = tflite::GetModel(g_modelo);
  if (modelo->version() != TFLITE_SCHEMA_VERSION) {
    erro = "versão do esquema do modelo incompatível";
    return false;
  }
  // Operadores de uma MobileNetV2 int8 com GAP + densa + softmax.
  // Se faltar algum, o erro no boot diz qual; a lista exata sai do Analyzer no notebook.
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddSub();
  resolver.AddPad();
  resolver.AddRelu6();
  resolver.AddMean();
  resolver.AddAveragePool2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  const uint32_t caps = (ARENA_EM_PSRAM ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL) | MALLOC_CAP_8BIT;
  Serial.printf("INFO,maior_bloco_livre_%s,%u\n", ARENA_EM_PSRAM ? "psram" : "interna",
                (unsigned)heap_caps_get_largest_free_block(caps));
  arena = (uint8_t*)heap_caps_aligned_alloc(16, ARENA_BYTES, caps);
  if (!arena) {
    erro = ARENA_EM_PSRAM ? "sem PSRAM para a tensor arena (PSRAM OPI está habilitada?)"
                          : "RAM interna sem bloco livre do tamanho da arena (use ARENA_EM_PSRAM 1)";
    return false;
  }
  interp = new tflite::MicroInterpreter(modelo, resolver, arena, ARENA_BYTES);
  if (interp->AllocateTensors() != kTfLiteOk) {
    erro = "AllocateTensors falhou (arena pequena ou operador faltando)";
    return false;
  }
  entrada = interp->input(0);
  saida = interp->output(0);
  if (entrada->type != kTfLiteInt8 || saida->type != kTfLiteInt8) {
    erro = "o modelo precisa ter entrada e saída int8";
    return false;
  }
  lado = entrada->dims->data[1];
  return true;
}

void inferencia_info() {
  Serial.printf("INFO,modelo_bytes,%u\n", g_modelo_len);
  Serial.printf("INFO,arena_local,%s\n", ARENA_EM_PSRAM ? "psram" : "interna");
  Serial.printf("INFO,arena_reservada,%u\n", (unsigned)ARENA_BYTES);
  Serial.printf("INFO,arena_usada,%u\n", (unsigned)interp->arena_used_bytes());
  Serial.printf("INFO,entrada,%dx%dx%d,escala=%.6f,zp=%d\n", entrada->dims->data[1], entrada->dims->data[2],
                entrada->dims->data[3], entrada->params.scale, entrada->params.zero_point);
  Serial.printf("INFO,saida,escala=%.6f,zp=%d\n", saida->params.scale, saida->params.zero_point);
  Serial.printf("INFO,limiar_maduro,%.2f\n", LIMIAR);
  Serial.printf("INFO,psram_livre,%u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  Serial.printf("INFO,ram_interna_livre,%u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
}

// RGB565 do esp32-camera: byte alto primeiro. Mesma expansão usada pelo esp32-camera ao gerar
// o JPEG do /capture (o dataset), sem replicar bits.
static inline void rgb565(const uint8_t* p, float& r, float& g, float& b) {
  uint8_t hb = p[0], lb = p[1];
  r = hb & 0xF8;
  g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
  b = (lb & 0x1F) << 3;
}

// Recorte quadrado central + redimensionamento bilinear com centros de meio pixel
// (igual ao tf.image.resize do treino e ao preproc.py do servidor) + quantização.
static void preparar(const camera_fb_t* fb, int8_t* dst) {
  const int W = fb->width, H = fb->height, L = min(W, H);
  const int x0 = (W - L) / 2, y0 = (H - L) / 2;
  const float esc = entrada->params.scale;
  const int zp = entrada->params.zero_point;
  const float passo = (float)L / lado;
  for (int i = 0; i < lado; i++) {
    float sy = constrain((i + 0.5f) * passo - 0.5f, 0.0f, (float)(L - 1));
    int ya = (int)sy, yb = min(ya + 1, L - 1);
    float wy = sy - ya;
    for (int j = 0; j < lado; j++) {
      float sx = constrain((j + 0.5f) * passo - 0.5f, 0.0f, (float)(L - 1));
      int xa = (int)sx, xb = min(xa + 1, L - 1);
      float wx = sx - xa;
      float c[4][3];
      const int ys[2] = {y0 + ya, y0 + yb}, xs[2] = {x0 + xa, x0 + xb};
      for (int k = 0; k < 4; k++) {
        const uint8_t* p = fb->buf + ((ys[k >> 1] * W) + xs[k & 1]) * 2;
        rgb565(p, c[k][0], c[k][1], c[k][2]);
      }
      for (int ch = 0; ch < 3; ch++) {
        float topo = c[0][ch] * (1 - wx) + c[1][ch] * wx;
        float base = c[2][ch] * (1 - wx) + c[3][ch] * wx;
        float v = topo * (1 - wy) + base * wy;
        int q = (int)lrintf(v / esc) + zp;   // lrintf: arredonda metade para par, como o numpy
        dst[(i * lado + j) * 3 + ch] = (int8_t)constrain(q, -128, 127);
      }
    }
  }
}

bool inferencia_classificar(const camera_fb_t* fb, Resultado& r) {
  if (fb->format != PIXFORMAT_RGB565) return false;
  uint32_t t0 = agora_us();
  preparar(fb, entrada->data.int8);
  uint32_t t1 = agora_us();
  soma_entrada = somar_entrada();   // fora da medida de tempo; antes do Invoke()
  uint32_t t1b = agora_us();
  if (interp->Invoke() != kTfLiteOk) return false;
  uint32_t t2 = agora_us();
  for (int k = 0; k < 2; k++) {
    r.bruto[k] = saida->data.int8[k];
    r.probs[k] = (saida->data.int8[k] - saida->params.zero_point) * saida->params.scale;
  }
  r.classe = r.probs[1] >= LIMIAR ? 1 : 0;
  r.conf = r.probs[r.classe];
  r.t_preproc_us = t1 - t0;
  r.t_inferencia_us = t2 - t1b;
  return true;
}

// A soma tem de ser feita ANTES do Invoke(): o TFLite Micro reaproveita a área da entrada
// para tensores intermediários, e depois da inferência ela já foi sobrescrita.
static long somar_entrada() {
  long s = 0;
  const int n = entrada->bytes;
  for (int i = 0; i < n; i++) s += entrada->data.int8[i];
  return s;
}
long inferencia_soma_entrada() { return soma_entrada; }

void inferencia_benchmark(const camera_fb_t* fb, int n) {
  preparar(fb, entrada->data.int8);
  uint32_t mn = UINT32_MAX, mx = 0;
  uint64_t soma = 0;
  for (int i = 0; i < n; i++) {
    uint32_t t0 = agora_us();
    interp->Invoke();
    uint32_t dt = agora_us() - t0;
    mn = min(mn, dt);
    mx = max(mx, dt);
    soma += dt;
  }
  Serial.printf("BENCH,invoke_ms,n=%d,min=%.2f,media=%.2f,max=%.2f\n", n, mn / 1000.0, soma / 1000.0 / n, mx / 1000.0);
}

void media_cor_centro(const camera_fb_t* fb) {
  const int W = fb->width, H = fb->height, m = 40;
  double s[3] = {0, 0, 0};
  int n = 0;
  for (int y = H / 2 - m; y < H / 2 + m; y++)
    for (int x = W / 2 - m; x < W / 2 + m; x++) {
      float r, g, b;
      rgb565(fb->buf + (y * W + x) * 2, r, g, b);
      s[0] += r; s[1] += g; s[2] += b; n++;
    }
  Serial.printf("COR,centro_80x80,R=%.0f,G=%.0f,B=%.0f\n", s[0] / n, s[1] / n, s[2] / n);
}
