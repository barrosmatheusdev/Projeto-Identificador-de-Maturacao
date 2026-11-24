# Firmware do classificador

`classificador/` é o sketch de operação: sensor → captura → decisão (no chip ou no servidor) → comporta, com registro de tempos no serial. A coleta do dataset continua sendo feita com o `Sketch_03_CameraWebServer` da Freenove e o `capturar.py`.

## Preparar

1. **Núcleo**: esp32 da Espressif **3.x** (testado na compilação com 3.3.12). Ele já traz o `esp-tflite-micro` com os kernels **ESP-NN** do ESP32-S3; não instale a biblioteca `TensorFlowLite_ESP32` (é antiga e não tem ESP-NN).
2. **Biblioteca**: `ESP32Servo` (Gerenciador de Bibliotecas).
3. **Modelo**: copie `modelo.h` e `modelo.cpp` gerados pelo notebook (`MyDrive/TCC/modelos/`) para `classificador/`.
4. **Wi-Fi**: copie `secrets_exemplo.h` para `secrets.h` e preencha (fica fora do versionamento).
5. **Placa**: ESP32S3 Dev Module · Flash Size 16MB · Partition Scheme *16M Flash (3MB APP/9.9MB FATFS)* · PSRAM *OPI PSRAM*. Gravar pela porta **UART**.

## Ligações (padrão do `config.h`, confirmar no pinout da Freenove)

| Peça | GPIO | Observação |
|---|---|---|
| Servo da comporta (sinal) | 14 | alimentação pela fonte 5 V, **GND comum** com o ESP32, capacitor 470–1000 µF perto do servo |
| Sensor IR (saída digital) | 21 | LOW = fruto presente (ajustável em `SENSOR_NIVEL_DETECTA`) |
| Servo do batente (opcional) | -1 | só no modo remoto; definir `PINO_BATENTE` |

A câmera usa os GPIOs 4–13 e 15–18, e a PSRAM usa 35–37. Nenhum desses pode ser usado para outra coisa. O clock da câmera (XCLK) é de **20 MHz**: com 10 MHz a imagem sai com listras.

## Comandos no monitor serial (115200, "Nova linha")

| Comando | O que faz | Para qual teste |
|---|---|---|
| `t` | ciclo completo sem sensor (captura → decisão → comporta) | bancada |
| `m` | alterna embarcado (Wi-Fi desligado) / remoto | R04, R06 |
| `f N` | define o número do próximo fruto | todos |
| `b` | 50 inferências seguidas, mostra mín./média/máx. | T03 (tempo de inferência) |
| `c` | média de R, G, B no centro da imagem | conferir cores (ver abaixo) |
| `x` | mesma imagem classificada no ESP32 e no servidor (int8 × int8) | verificação da implementação |
| `s` | move a comporta para maduro, verde e lateral | calibrar ângulos |
| `i` | memória, arena usada, quantização | T05 (R05) |

Cada fruto gera uma linha `LOG,...` com os tempos em ms: sensor→captura, pré-processamento, inferência, JPEG, rede, servidor, **captura→comporta** e total. O `registrar_serial.py` grava essas linhas em `testes/log_*.csv`.

## Primeiros testes (sem tomate)

1. Grave e abra o monitor serial. As linhas `INFO,arena_usada` e `INFO,modelo_bytes` dão os números do R05.
2. `b` mostra o tempo de inferência no chip.
3. `c` com um objeto **vermelho** no centro: R tem que sair bem maior que B. Se sair o contrário, a ordem dos bytes do RGB565 está trocada. Me avise antes de seguir.
4. `x` com o servidor rodando o **modelo int8 com kernels de referência** (`.venv\Scripts\python.exe app.py --modelo ../modelos/embarcado_etapa1_int8.tflite --referencia`): `soma_entrada` e os valores `int8=[..]` do ESP32 e `saida_int8` do servidor devem ser **iguais**, e foram nos testes de 2026-11-14. Sem `--referencia`, o PC usa o XNNPACK e a saída pode diferir alguns pontos.
5. `s` com o servo ligado, para calibrar `ANG_MADURO`, `ANG_VERDE` e `ANG_FALHA` no `config.h`.
