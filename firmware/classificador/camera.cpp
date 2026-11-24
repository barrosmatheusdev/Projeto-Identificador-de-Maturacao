#include <Arduino.h>
#include "camera.h"
#include "config.h"
#include "pinos_camera.h"

bool camera_iniciar() {
  camera_config_t c = {};
  // Igual ao Sketch_03 (canal 0 / timer 0). O servo fica preso ao timer 1 no setup()
  // (ESP32PWM::allocateTimer), então não mexe no clock da câmera.
  // Com canal 7 / timer 3 e 2 quadros no buffer a imagem saiu lavada (diário, entrada 25).
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;
  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  // 20 MHz: com os 10 MHz do Sketch_03 a OV5640 desta placa passou a gerar só listras
  // verticais (diário, entrada 25). O dataset também é coletado em 20 MHz (capturar.py).
  c.xclk_freq_hz = 20000000;
  c.frame_size = FRAMESIZE_VGA;
  c.pixel_format = PIXFORMAT_RGB565;
  c.grab_mode = CAMERA_GRAB_LATEST;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.jpeg_quality = 12;
  c.fb_count = 1;   // igual ao Sketch_03

  if (esp_camera_init(&c) != ESP_OK) return false;
  sensor_t* s = esp_camera_sensor_get();
  s->set_vflip(s, 1);
  s->set_brightness(s, 1);
  s->set_saturation(s, -1);
  // deixa o controle automático de exposição e branco estabilizar
  for (int i = 0; i < 10; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
  return true;
}

camera_fb_t* camera_capturar() {
#if DESCARTAR_QUADRO
  camera_fb_t* velho = esp_camera_fb_get();
  if (velho) esp_camera_fb_return(velho);
#endif
  return esp_camera_fb_get();
}
