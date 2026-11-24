#pragma once
#include "esp_camera.h"

// Mesma configuração do Sketch_03_CameraWebServer usado na coleta do dataset:
// VGA, RGB565, XCLK 10 MHz, vflip 1, brilho +1, saturação -1.
bool camera_iniciar();

// Devolve um quadro novo (descarta o que estava no buffer se DESCARTAR_QUADRO = 1).
camera_fb_t* camera_capturar();
