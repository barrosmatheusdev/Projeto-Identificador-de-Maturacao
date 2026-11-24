#!/usr/bin/env python3
"""Captura de imagens de tomates a partir do CameraWebServer (ESP32-S3 + OV5640).

Fluxo por fruto: o script pergunta o identificador e a classe (verde/maduro),
depois baixa uma foto do endpoint /capture a cada INTERVALO segundos, uma por
angulo. Entre uma foto e outra voce gira o fruto sobre o proprio eixo.

As fotos vao para  <saida>/<classe>/dia<DD>_<fruto>_a<K>.jpg  e cada uma ganha
uma linha em  <saida>/metadata.csv  (usado depois para dividir treino,
validacao e teste POR FRUTO).

Uso:
    python3 capturar.py --ip 192.168.0.50 --dia 1 --vga
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

CLASSES = {"v": "verde", "m": "maduro"}


def baixar(url, tentativas=2, timeout=8):
    """Baixa um JPEG e confere o cabecalho. Devolve bytes ou None."""
    for _ in range(tentativas):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                dados = r.read()
            if dados[:2] == b"\xff\xd8" and len(dados) > 1000:
                return dados
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    return None


FRAMESIZE_VGA = 10   # esp32-camera do nucleo 3.x: 8 e CIF (400x296), 10 e VGA (640x480)


def definir_vga(ip):
    """Pede ao CameraWebServer a resolucao VGA e confere pelo /status."""
    try:
        urllib.request.urlopen(f"http://{ip}/control?var=framesize&val={FRAMESIZE_VGA}", timeout=5).read()
        time.sleep(0.5)
        status = json.loads(urllib.request.urlopen(f"http://{ip}/status", timeout=5).read())
        if status.get("framesize") != FRAMESIZE_VGA:
            sys.exit(f"Resolucao ficou em framesize={status.get('framesize')}, esperado {FRAMESIZE_VGA} (VGA).")
        print("Resolucao ajustada para VGA (640x480).")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        sys.exit(f"Nao consegui ajustar a resolucao ({e}).")


def tamanho_jpeg(dados):
    """Largura e altura lidas do cabecalho JPEG (marcador SOF), sem bibliotecas."""
    i = 2
    while i + 9 < len(dados):
        if dados[i] != 0xFF:
            return None
        marca, tam = dados[i + 1], int.from_bytes(dados[i + 2:i + 4], "big")
        if marca in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(dados[i + 7:i + 9], "big"), int.from_bytes(dados[i + 5:i + 7], "big")
        i += 2 + tam
    return None


def definir_xclk(ip, mhz):
    """Ajusta o clock da camera. Com 10 MHz (padrao do Sketch_03) a OV5640 desta placa
    gera so listras verticais; com 20 MHz a imagem sai normal. O CameraWebServer volta
    para 10 MHz a cada reinicio, entao o ajuste e feito toda vez."""
    try:
        urllib.request.urlopen(f"http://{ip}/xclk?xclk={mhz}", timeout=5).read()
        time.sleep(1.0)
        status = json.loads(urllib.request.urlopen(f"http://{ip}/status", timeout=5).read())
        atual = status.get("xclk")
        if atual != mhz:
            sys.exit(f"XCLK ficou em {atual} MHz, esperado {mhz}. Ajuste na pagina e rode de novo.")
        print(f"XCLK ajustado para {mhz} MHz.")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        sys.exit(f"Nao consegui ajustar o XCLK ({e}). Sem isso as fotos saem com listras.")


def abrir_csv(caminho):
    novo = not os.path.exists(caminho)
    f = open(caminho, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if novo:
        w.writerow(["arquivo", "fruto", "classe", "dia", "angulo", "data_hora"])
    return f, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", required=True, help="IP do ESP32 (aparece no monitor serial)")
    ap.add_argument("--dia", type=int, required=True, help="numero do dia da coleta (1 a 10)")
    ap.add_argument("--angulos", type=int, default=5, help="fotos por fruto (padrao 5)")
    ap.add_argument("--intervalo", type=float, default=2.0, help="segundos entre fotos (padrao 2)")
    ap.add_argument("--saida", default="dataset", help="pasta de saida (padrao: dataset)")
    ap.add_argument("--vga", action="store_true", help="ajusta a camera para VGA antes de comecar")
    ap.add_argument("--xclk", type=int, default=20, help="clock da camera em MHz (padrao 20, igual ao firmware)")
    a = ap.parse_args()

    url = f"http://{a.ip}/capture"
    definir_xclk(a.ip, a.xclk)
    if a.vga:
        definir_vga(a.ip)

    teste = baixar(url)
    if teste is None:
        sys.exit(f"Nao consegui baixar imagem de {url}. Confira o IP e se o CameraWebServer esta rodando.")
    dim = tamanho_jpeg(teste)
    if a.vga and dim != (640, 480):
        sys.exit(f"A camera esta entregando {dim}, e nao 640x480. Ajuste a resolucao e rode de novo.")
    print(f"Camera respondendo ({dim[0]}x{dim[1]}).\n" if dim else "Camera respondendo.\n")

    for c in CLASSES.values():
        os.makedirs(os.path.join(a.saida, c), exist_ok=True)
    f, w = abrir_csv(os.path.join(a.saida, "metadata.csv"))

    total = 0
    try:
        while True:
            fruto = input("Numero do fruto (ou Enter para terminar): ").strip()
            if not fruto:
                break
            fruto = fruto.zfill(2)
            sigla = ""
            while sigla not in CLASSES:
                sigla = input("  Classe agora [v = verde (<50% vermelho) / m = maduro (>=50%)]: ").strip().lower()
            classe = CLASSES[sigla]
            input(f"  Fruto {fruto} ({classe}) posicionado sob a camera. Enter para comecar...")

            for k in range(1, a.angulos + 1):
                dados = baixar(url)
                if dados is None:
                    print(f"  angulo {k}: falhou, tente de novo esse fruto depois.")
                    continue
                nome = f"dia{a.dia:02d}_fruto{fruto}_a{k}.jpg"
                caminho = os.path.join(a.saida, classe, nome)
                with open(caminho, "wb") as img:
                    img.write(dados)
                w.writerow([os.path.join(classe, nome), fruto, classe, a.dia, k,
                            datetime.now().isoformat(timespec="seconds")])
                f.flush()
                total += 1
                print(f"  angulo {k}/{a.angulos} salvo ({len(dados)//1024} KB)", end="")
                if k < a.angulos:
                    print(f" -> gire o fruto, proxima foto em {a.intervalo:.0f} s")
                    time.sleep(a.intervalo)
                else:
                    print()
            print()
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        f.close()
    print(f"Fim: {total} imagens salvas em '{a.saida}'.")


if __name__ == "__main__":
    main()
