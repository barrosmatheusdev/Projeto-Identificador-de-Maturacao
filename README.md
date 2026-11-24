# Identificador de maturação de tomates (ESP32-S3)

Código do TCC de Ciência da Computação (FAM), modalidade Protótipo: classificação de tomates em verde ou maduro com ESP32-S3 e câmera OV5640.

- `firmware/`: sketch Arduino (câmera, inferência com TFLite Micro, modo remoto e servo). Copie `secrets_exemplo.h` para `secrets.h`.
- `treino/`: notebook do Colab (MobileNetV2, quantização int8) e o script que o gera.
- `servidor/`: servidor Flask da inferência remota.
- `coleta/`: captura de imagens e script do teste com a placa.
- `testes/preliminar_etapa1.csv`: registro bruto do teste preliminar com tomates reais.

Rampa, servo e sensor não foram montados nesta versão. O modelo foi treinado só com as bases públicas [Mputu et al. (2023)](https://data.mendeley.com/datasets/x4s2jz55dx/1) e [Khatun et al. (2023)](https://data.mendeley.com/datasets/s42kpg8h37/1), ambas CC BY 4.0.
