"""Pré-processamento igual ao do notebook de treino (treino/treino_maturacao.ipynb).

JPEG -> RGB -> recorte quadrado central -> redimensiona (bilinear, mesmo cálculo do
tf.image.resize, com centros de meio pixel e sem antialias) -> float32 0..255.
A normalização para [-1, 1] está dentro do modelo, então não é feita aqui.
"""
import io

import numpy as np
from PIL import Image


def decodificar(jpeg: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg)) as img:
        return np.asarray(img.convert("RGB"))


def decodificar_rgb565(dados: bytes, largura: int, altura: int) -> np.ndarray:
    """Quadro cru do esp32-camera (byte alto primeiro), com a mesma expansão para 8 bits
    usada pelo firmware e pelo esp32-camera ao gerar JPEG (sem replicar bits)."""
    px = np.frombuffer(dados, dtype=np.uint8)
    if px.size != largura * altura * 2:
        raise ValueError(f"esperava {largura * altura * 2} bytes, recebi {px.size}")
    hb = px[0::2].astype(np.uint16)
    lb = px[1::2].astype(np.uint16)
    r = hb & 0xF8
    g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3)
    b = (lb & 0x1F) << 3
    return np.stack([r, g, b], axis=-1).reshape(altura, largura, 3).astype(np.uint8)


def recorte_central(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    lado = min(h, w)
    y, x = (h - lado) // 2, (w - lado) // 2
    return img[y:y + lado, x:x + lado]


def redimensionar_bilinear(img: np.ndarray, tam: int) -> np.ndarray:
    """Equivale a tf.image.resize(img, (tam, tam), method='bilinear')."""
    h, w = img.shape[:2]
    img = img.astype(np.float32)
    ys = np.clip((np.arange(tam) + 0.5) * h / tam - 0.5, 0, h - 1)
    xs = np.clip((np.arange(tam) + 0.5) * w / tam - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    topo = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    base = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return topo * (1 - wy) + base * wy


def preparar(jpeg: bytes, tam: int) -> np.ndarray:
    return redimensionar_bilinear(recorte_central(decodificar(jpeg)), tam)


def preparar_rgb565(dados: bytes, largura: int, altura: int, tam: int) -> np.ndarray:
    return redimensionar_bilinear(recorte_central(decodificar_rgb565(dados, largura, altura)), tam)
