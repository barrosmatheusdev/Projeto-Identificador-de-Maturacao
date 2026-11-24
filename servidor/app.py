#!/usr/bin/env python3
"""Servidor de inferência remota (modo remoto do protótipo).

O ESP32 envia o JPEG capturado e recebe a classe em JSON. Roda no Windows (testes)
e no Ubuntu do Orange Pi Zero 3 sem mudança de código.

Uso:
    python app.py --modelo ../modelos/servidor_etapa1_float.tflite
    python app.py --modelo ../modelos/servidor_etapa1.keras
    python app.py --falso          # sem modelo, só para testar a comunicação

Endpoints:
    POST /classificar[?fruto=07]   corpo = JPEG cru (Content-Type: image/jpeg)
                                   ou multipart com o campo "imagem"
    GET  /saude                    informações do modelo carregado

Cada requisição vira uma linha em registros/requisicoes.csv com os tempos medidos.
"""
import argparse
import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

from preproc import preparar, preparar_rgb565

CLASSES = ["verde", "maduro"]
AQUI = Path(__file__).resolve().parent


def _interpreter():
    """Interpretador TFLite mais leve disponível (no Orange Pi, ai-edge-litert)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    import tensorflow as tf
    return tf.lite.Interpreter


class Modelo:
    def __init__(self, caminho=None, falso=False, referencia=False):
        self.trava = threading.Lock()   # interpretador TFLite não é thread-safe
        self.ultima_int8 = None         # saída int8 crua (só para modelos quantizados)
        self.ultima_soma = None         # soma dos int8 da entrada (só para modelos quantizados)
        if falso:
            self.descricao, self.tam = "falso (heurística de cor, só para testar a comunicação)", 96
            self._prever = self._prever_falso
            return
        caminho = Path(caminho)
        if caminho.suffix == ".keras":
            import keras
            self._m = keras.models.load_model(caminho)
            self.tam = int(self._m.input_shape[1])
            self._prever = lambda x: np.asarray(self._m(x[None], training=False))[0]
        elif caminho.suffix == ".tflite":
            Interp = _interpreter()
            if referencia:
                # Kernels de referência, sem o XNNPACK. Dão saída int8 idêntica à do ESP32 (ESP-NN);
                # o XNNPACK, ligado por padrão no PC, arredonda diferente (diário, entrada 26).
                try:
                    from ai_edge_litert.interpreter import OpResolverType
                except ImportError:
                    import tensorflow as tf
                    OpResolverType = tf.lite.experimental.OpResolverType
                it = Interp(model_path=str(caminho), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
            else:
                it = Interp(model_path=str(caminho))
            it.allocate_tensors()
            ent, sai = it.get_input_details()[0], it.get_output_details()[0]
            self.tam = int(ent["shape"][1])

            def prever(x):
                if ent["dtype"] == np.int8:              # modelo do ESP32 rodando no PC
                    esc, zp = ent["quantization"]
                    x = np.clip(np.round(x / esc) + zp, -128, 127).astype(np.int8)
                    self.ultima_soma = int(x.astype(np.int64).sum())   # conferência com o ESP32
                it.set_tensor(ent["index"], x[None].astype(ent["dtype"]))
                it.invoke()
                y = it.get_tensor(sai["index"])[0]
                if sai["dtype"] == np.int8:
                    self.ultima_int8 = [int(v) for v in y]
                    esc, zp = sai["quantization"]
                    y = (y.astype(np.float32) - zp) * esc
                return y
            self._prever = prever
        else:
            raise SystemExit(f"Formato não suportado: {caminho.suffix} (use .keras ou .tflite)")
        self.descricao = caminho.name + (" (kernels de referência)" if referencia else "")

    @staticmethod
    def _prever_falso(x):
        r, g = x[..., 0], x[..., 1]
        frac = float(np.mean(r > g + 30))     # fração de pixels "vermelhos"
        return np.array([1 - frac, frac], dtype=np.float32)

    def prever(self, x):
        with self.trava:
            y = self._prever(x)
            return y, self.ultima_int8


def criar_app(modelo, pasta_reg, salvar_imagens=False):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    pasta_reg.mkdir(parents=True, exist_ok=True)
    arq_csv = pasta_reg / "requisicoes.csv"
    trava_csv = threading.Lock()
    if not arq_csv.exists():
        with open(arq_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["data_hora", "cliente", "fruto", "bytes", "classe", "confianca",
                                    "t_preproc_ms", "t_inferencia_ms", "t_servidor_ms", "modelo"])

    @app.get("/saude")
    def saude():
        return jsonify(modelo=modelo.descricao, entrada=modelo.tam, classes=CLASSES)

    @app.post("/classificar")
    def classificar():
        t0 = time.perf_counter()
        arq = request.files.get("imagem")
        jpeg = arq.read() if arq else request.get_data()
        fruto = request.args.get("fruto", "")
        bruto = request.content_type == "image/x-rgb565"   # comando "x" do firmware
        if not bruto and (len(jpeg) < 2 or jpeg[:2] != b"\xff\xd8"):
            return jsonify(erro="corpo não é um JPEG"), 400
        try:
            if bruto:
                x = preparar_rgb565(jpeg, int(request.args["w"]), int(request.args["h"]), modelo.tam)
            else:
                x = preparar(jpeg, modelo.tam)
        except Exception as e:  # JPEG truncado vindo do ESP32, por exemplo
            return jsonify(erro=f"falha ao decodificar: {e}"), 400
        t1 = time.perf_counter()
        probs, saida_int8 = modelo.prever(x)
        t2 = time.perf_counter()
        idx = int(np.argmax(probs))
        resp = {
            "classe": CLASSES[idx],
            "indice": idx,
            "confianca": round(float(probs[idx]), 4),
            "probs": [round(float(p), 4) for p in probs],
            "t_preproc_ms": round((t1 - t0) * 1000, 2),
            "t_inferencia_ms": round((t2 - t1) * 1000, 2),
            "t_servidor_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        if saida_int8 is not None:
            resp["saida_int8"] = saida_int8
            resp["soma_entrada"] = modelo.ultima_soma
        agora = datetime.now()
        if salvar_imagens:
            (pasta_reg / "imagens").mkdir(exist_ok=True)
            nome = agora.strftime("%Y%m%d_%H%M%S_%f") + (f"_f{fruto}" if fruto else "") + f"_{resp['classe']}.jpg"
            (pasta_reg / "imagens" / nome).write_bytes(jpeg)
        with trava_csv, open(arq_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([agora.isoformat(timespec="milliseconds"), request.remote_addr, fruto,
                                    len(jpeg), resp["classe"], resp["confianca"], resp["t_preproc_ms"],
                                    resp["t_inferencia_ms"], resp["t_servidor_ms"], modelo.descricao])
        return jsonify(resp)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--modelo", help="arquivo .keras ou .tflite")
    g.add_argument("--falso", action="store_true", help="sem modelo, responde por heurística de cor")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--porta", type=int, default=5000)
    ap.add_argument("--registros", default=str(AQUI / "registros"), help="pasta do CSV de tempos")
    ap.add_argument("--salvar-imagens", action="store_true", help="guarda cada imagem recebida")
    ap.add_argument("--referencia", action="store_true",
                    help="kernels de referência do TFLite (sem XNNPACK): use para comparar com o ESP32 (comando x)")
    a = ap.parse_args()

    modelo = Modelo(a.modelo, falso=a.falso, referencia=a.referencia)
    # aquecimento: a primeira inferência é sempre mais lenta e não deve entrar nas medidas
    for _ in range(3):
        modelo.prever(np.zeros((modelo.tam, modelo.tam, 3), np.float32))
    print(f"Modelo: {modelo.descricao} | entrada {modelo.tam}x{modelo.tam} | pid {os.getpid()}")
    app = criar_app(modelo, Path(a.registros), a.salvar_imagens)
    app.run(host=a.host, port=a.porta, threaded=True)


if __name__ == "__main__":
    main()
