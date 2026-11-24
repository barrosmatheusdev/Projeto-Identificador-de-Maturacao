#!/usr/bin/env python3
"""Cliente de teste do servidor: faz o papel do ESP32 enquanto o firmware não está pronto.

Uso:
    python cliente_teste.py --url http://127.0.0.1:5000 foto.jpg -n 50
    python cliente_teste.py --url http://192.168.0.10:5000 ../dataset/maduro

Com uma imagem e -n, repete o envio e mostra a distribuição do tempo de ida e volta.
Com uma pasta, classifica cada imagem uma vez.
"""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def enviar(url, jpeg, timeout):
    req = urllib.request.Request(f"{url}/classificar", data=jpeg, method="POST",
                                 headers={"Content-Type": "image/jpeg"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return resp, (time.perf_counter() - t0) * 1000


def resumo(nome, v):
    v = sorted(v)
    p95 = v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]
    print(f"{nome:16s} min {v[0]:7.1f}  média {statistics.mean(v):7.1f}  p95 {p95:7.1f}  máx {v[-1]:7.1f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("alvo", help="arquivo JPEG ou pasta com JPEGs")
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("-n", type=int, default=1, help="repetições (só com um arquivo)")
    ap.add_argument("--timeout", type=float, default=2.0, help="limite do R06, em segundos")
    a = ap.parse_args()

    alvo = Path(a.alvo)
    arquivos = sorted(p for p in alvo.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"}) if alvo.is_dir() else [alvo] * a.n
    ida_volta, servidor, falhas = [], [], 0
    for arq in arquivos:
        try:
            resp, ms = enviar(a.url, arq.read_bytes(), a.timeout)
        except Exception as e:
            falhas += 1
            print(f"{arq.name}: sem resposta em {a.timeout} s ({e})")
            continue
        ida_volta.append(ms)
        servidor.append(resp["t_servidor_ms"])
        if alvo.is_dir() or a.n == 1:
            print(f"{arq.name}: {resp['classe']} ({resp['confianca']:.2f})  ida e volta {ms:.1f} ms")
    if len(ida_volta) > 1:
        resumo("ida e volta", ida_volta)
        resumo("no servidor", servidor)
    print(f"{len(ida_volta)} respostas, {falhas} falhas")


if __name__ == "__main__":
    main()
