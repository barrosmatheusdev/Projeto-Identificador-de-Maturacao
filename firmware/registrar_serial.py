#!/usr/bin/env python3
"""Monitor serial que grava os registros do firmware em arquivo (R07).

Substitui o monitor serial da Arduino IDE durante os testes: mostra tudo na tela,
deixa digitar os comandos do firmware (t, m, b, x, ...) e grava:
  - linhas LOG,...      -> <saida>/log_<data>.csv   (uma linha por fruto)
  - INFO/BENCH/COMPARA/COR/ERRO -> <saida>/eventos_<data>.txt

Uso:
    pip install pyserial
    python registrar_serial.py --porta COM5 --teste T03
(feche o monitor serial da Arduino IDE antes; só um programa pode abrir a porta)
"""
import argparse
import sys
import threading
from datetime import datetime
from pathlib import Path

import serial

CAB = ("fruto,modo,classe,confianca,p_verde,p_maduro,int8_verde,int8_maduro,t_sensor_captura_ms,"
       "t_preproc_ms,t_inferencia_ms,t_jpeg_ms,t_rede_ms,t_servidor_ms,t_captura_comporta_ms,t_total_ms,status")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--porta", required=True, help="ex.: COM5 (Windows) ou /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--teste", default="", help="rótulo do teste, entra no nome do arquivo (ex.: T03)")
    ap.add_argument("--saida", default=str(Path(__file__).resolve().parent.parent / "testes"))
    a = ap.parse_args()

    pasta = Path(a.saida)
    pasta.mkdir(parents=True, exist_ok=True)
    sufixo = (a.teste + "_" if a.teste else "") + datetime.now().strftime("%Y%m%d_%H%M%S")
    arq_log = pasta / f"log_{sufixo}.csv"
    arq_ev = pasta / f"eventos_{sufixo}.txt"

    s = serial.Serial(a.porta, a.baud, timeout=0.2)
    print(f"Gravando em {arq_log.name} e {arq_ev.name}. Digite comandos e Enter; Ctrl+C sai.\n")

    def teclado():
        for linha in sys.stdin:
            s.write(linha.encode())

    threading.Thread(target=teclado, daemon=True).start()
    with open(arq_log, "w", encoding="utf-8", newline="") as flog, open(arq_ev, "w", encoding="utf-8") as fev:
        flog.write("data_hora_pc," + CAB + "\n")
        try:
            while True:
                bruto = s.readline()
                if not bruto:
                    continue
                lin = bruto.decode("utf-8", errors="replace").rstrip()
                print(lin)
                agora = datetime.now().isoformat(timespec="milliseconds")
                if lin.startswith("LOG,"):
                    flog.write(f"{agora},{lin[4:]}\n")
                    flog.flush()
                elif lin.split(",", 1)[0] in {"INFO", "BENCH", "COMPARA", "COR", "ERRO"}:
                    fev.write(f"{agora} {lin}\n")
                    fev.flush()
        except KeyboardInterrupt:
            pass
    s.close()
    print(f"\nArquivos em {pasta}")


if __name__ == "__main__":
    main()
