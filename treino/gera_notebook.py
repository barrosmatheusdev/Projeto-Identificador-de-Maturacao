import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Treino do classificador de maturação (verde / maduro)

Notebook do TCC (seções 3.3 e 3.4, Quadro 4). Faz, nesta ordem:

1. **Etapa 1 – pré-treino** com Mputu et al. (2023) + Khatun et al. (2023) juntos (Khatun só com imagens originais), cada um dividido 70/15/15: só a cabeça, base congelada, Adam 1e-3, até 20 épocas, paciência 5. Métricas no teste de cada conjunto público.
2. **Etapa 2 – ajuste fino** com o conjunto próprio (pula sozinho se a pasta ainda não existir): cabeça + últimos blocos, Adam 1e-5, até 30 épocas, paciência 5, divisão 70/15/15 **por fruto**.
3. **Generalização para a câmera do ESP32**: o modelo da etapa 1, sem ajuste fino, é avaliado no teste próprio e comparado com o modelo ajustado.

Por que os dois públicos entram no pré-treino: o Mputu tem só 93 imagens verdes distintas (as 800 da pasta `Unripe` são cópias dessas 93), contra 797 maduras, então sozinho ele ensinaria pouco sobre a classe verde.
4. **Quantização int8** do modelo embarcado (≈200 imagens de calibração), avaliação do `.tflite` no PC e exportação para vetor C (`modelo.h` / `modelo.cpp`).

Dois modelos: **embarcado** (MobileNetV2 α = 0,35, 96×96) e **servidor** (MobileNetV2 α = 1,0, 224×224).

**Antes de rodar**
- Ambiente de execução → Alterar tipo → **GPU**.
- Nas páginas do Mendeley Data clique em *Download All* e salve no Drive em `MyDrive/TCC/dados/` com os nomes **`mputu.zip`** e **`khatun.zip`**:
  - Mputu: https://data.mendeley.com/datasets/x4s2jz55dx/1
  - Khatun: https://data.mendeley.com/datasets/s42kpg8h37/1
- (Depois da coleta) suba a pasta `dataset/` gerada pelo `capturar.py` para `MyDrive/TCC/dataset/`.

**Pré-processamento** (o firmware precisa fazer igual): recorte quadrado central → redimensiona para o tamanho de entrada → pixels 0–255. A normalização para [-1, 1] está **dentro** do modelo, então no ESP32 a entrada int8 é praticamente `pixel - 128` (o valor exato da escala e do ponto zero é impresso na seção de quantização).
""")

code(r"""
import os, re, csv, json, hashlib, zipfile, random, time
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print('TensorFlow', tf.__version__, '| Keras', keras.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))

SEED = 42
keras.utils.set_random_seed(SEED)

USAR_DRIVE = True
if USAR_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    RAIZ = Path('/content/drive/MyDrive/TCC')
else:
    RAIZ = Path('/content/TCC')

DADOS_ZIP = RAIZ / 'dados'        # mputu.zip e khatun.zip
PROPRIO   = RAIZ / 'dataset'      # saída do capturar.py (metadata.csv + verde/ + maduro/)
SAIDA     = RAIZ / 'modelos'      # modelos, divisões e métricas
TRAB      = Path('/content/trab') # extração local, mais rápida que ler do Drive
for p in (DADOS_ZIP, SAIDA, TRAB):
    p.mkdir(parents=True, exist_ok=True)

CLASSES = ['verde', 'maduro']     # índices 0 e 1; "maduro" é a classe positiva (R02)
MODELOS = {
    'embarcado': dict(alpha=0.35, tam=96),
    'servidor':  dict(alpha=1.0,  tam=224),
}
TREINAR = ['embarcado', 'servidor']   # tire um da lista para rodar só o outro
LOTE = 32
AUTOTUNE = tf.data.AUTOTUNE
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

ARQ_RESULTADOS = SAIDA / 'resultados.json'
RESULTADOS = json.loads(ARQ_RESULTADOS.read_text()) if ARQ_RESULTADOS.exists() else {}
def registrar(chave, valor):
    RESULTADOS[chave] = valor
    ARQ_RESULTADOS.write_text(json.dumps(RESULTADOS, indent=2, ensure_ascii=False))
""")

md(r"""
## 1. Extração e inspeção dos conjuntos públicos

A tabela mostra cada pasta e quantas imagens tem. **Confira antes de seguir**: é por ela que você vê se o mapeamento da seção 2 pegou as pastas certas.
""")

code(r"""
import shutil, subprocess

def garantir_unrar():
    if shutil.which('unrar'):
        return 'unrar'
    if shutil.which('apt-get'):   # Colab / Ubuntu
        subprocess.run('apt-get -qq install -y unrar > /dev/null', shell=True, check=False)
        if shutil.which('unrar'):
            return 'unrar'
    if shutil.which('7z'):
        return '7z'
    raise RuntimeError('Preciso do "unrar" ou do "7z" para abrir os .rar (no Colab: !apt-get install -y unrar)')

def extrair_arquivo(arq, destino):
    destino.mkdir(parents=True, exist_ok=True)
    if arq.suffix.lower() == '.zip':
        with zipfile.ZipFile(arq) as z:
            z.extractall(destino)
    else:   # .rar (e .7z, se aparecer)
        prog = garantir_unrar() if arq.suffix.lower() == '.rar' else '7z'
        cmd = ['unrar', 'x', '-o+', '-inul', str(arq), str(destino) + '/'] if prog == 'unrar' \
              else ['7z', 'x', '-y', f'-o{destino}', str(arq)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'Falha ao extrair {arq.name}: {r.stderr[-500:]}')

def extrair(zip_path, destino):
    destino = Path(destino)
    if destino.exists() and any(destino.iterdir()):
        print(f'{destino.name}: já extraído')
        return destino
    if not Path(zip_path).exists():
        raise FileNotFoundError(f'Não achei {zip_path}. Veja as instruções no topo do notebook.')
    extrair_arquivo(Path(zip_path), destino)
    # o "Download All" do Mendeley traz o conjunto compactado dentro do zip (zip ou rar)
    while True:
        internos = [p for p in destino.rglob('*') if p.is_file() and p.suffix.lower() in {'.zip', '.rar', '.7z'}]
        if not internos:
            break
        for arq in internos:
            print('extraindo', arq.relative_to(destino))
            extrair_arquivo(arq, arq.with_suffix(''))
            arq.unlink()
    return destino

def imagens(raiz):
    return sorted(f for f in Path(raiz).rglob('*') if f.suffix.lower() in EXT)

def arvore(raiz):
    cont = {}
    for f in imagens(raiz):
        d = f.parent.relative_to(raiz)
        cont[d] = cont.get(d, 0) + 1
    for d, n in sorted(cont.items(), key=lambda x: str(x[0])):
        print(f'{n:6d}  {d}')
    print(f'{sum(cont.values()):6d}  TOTAL')

DIR_MPUTU  = extrair(DADOS_ZIP / 'mputu.zip',  TRAB / 'mputu')
DIR_KHATUN = extrair(DADOS_ZIP / 'khatun.zip', TRAB / 'khatun')
print('\n=== Mputu ==='); arvore(DIR_MPUTU)
print('\n=== Khatun ==='); arvore(DIR_KHATUN)
""")

md(r"""
## 2. Mapeamento das pastas para verde / maduro

A classe sai das **palavras inteiras** do caminho (pastas + nome do arquivo, em minúsculas), então `ripe` não casa com `unripe`.
Ficam de fora: rejeitado/podre/defeito (fora do escopo) e imagens aumentadas (Khatun só com originais, seção 3.3).
Se a tabela de baixo mostrar alguma pasta no lugar errado, ajuste os conjuntos de palavras e rode de novo.
""")

code(r"""
PALAVRAS = {
    'verde':  {'unripe', 'immature', 'green'},
    'maduro': {'ripe', 'mature'},
}
EXCLUIR = {'reject', 'rejected', 'rotten', 'defect', 'defected', 'damaged', 'spoiled',
           'augment', 'augmented', 'augmentation', 'aug'}

def classe_do_caminho(rel):
    tokens = set(re.split(r'[^a-z]+', str(rel).lower()))
    if tokens & EXCLUIR:
        return None, 'excluída'
    achou = [c for c, pal in PALAVRAS.items() if tokens & pal]
    if len(achou) != 1:
        return None, 'ambígua' if achou else 'sem rótulo'
    return CLASSES.index(achou[0]), achou[0]

def md5(f):
    return hashlib.md5(Path(f).read_bytes()).hexdigest()

def listar(raiz, nome):
    # 1ª passada: rótulo e hash de TODAS as imagens, inclusive as excluídas
    resumo, por_hash = {}, {}
    for f in imagens(raiz):
        rel = f.relative_to(raiz)
        rot, motivo = classe_do_caminho(rel)
        chave = (str(rel.parent), motivo)
        resumo[chave] = resumo.get(chave, 0) + 1
        if motivo == 'sem rótulo':   # pastas sem classe de maturação (ex.: Two Classes/Healthy)
            continue
        por_hash.setdefault(md5(f), []).append((f, motivo))
    # 2ª passada: uma imagem por conteúdo; conteúdo com rótulos diferentes sai (rótulo contraditório)
    itens, copias, conflitos = [], 0, 0
    for h, lst in por_hash.items():
        motivos = {m for _, m in lst}
        if len(motivos) > 1:
            conflitos += 1
            continue
        m = motivos.pop()
        if m not in CLASSES:
            continue
        copias += len(lst) - 1
        itens.append((sorted(f for f, _ in lst)[0], CLASSES.index(m)))
    itens.sort()
    print(f'=== {nome}: pasta -> destino ===')
    for (d, m), n in sorted(resumo.items()):
        print(f'{n:6d}  {m:10s}  {d}')
    cont = np.bincount([r for _, r in itens], minlength=2)
    print(f'Usadas: {len(itens)} distintas (verde {cont[0]}, maduro {cont[1]}); '
          f'cópias removidas: {copias}; conteúdos com rótulo contraditório removidos: {conflitos}\n')
    registrar(f'limpeza_{nome.lower()}', {'distintas': len(itens), 'verde': int(cont[0]), 'maduro': int(cont[1]),
                                          'copias_removidas': copias, 'rotulo_contraditorio': conflitos})
    return itens, set(por_hash)

ITENS_MPUTU, HASH_MPUTU = listar(DIR_MPUTU, 'Mputu')
ITENS_KHATUN, _ = listar(DIR_KHATUN, 'Khatun')

# a mesma foto não pode estar nos dois conjuntos (cairia em treino de um e teste do outro)
antes = len(ITENS_KHATUN)
ITENS_KHATUN = [(f, r) for f, r in ITENS_KHATUN if md5(f) not in HASH_MPUTU]
print(f'Khatun: {antes - len(ITENS_KHATUN)} imagens também presentes no Mputu foram retiradas.')
""")

md(r"""
## 3. Divisão 70/15/15

A divisão é feita **por grupo**: todas as imagens de um grupo caem na mesma partição. No conjunto próprio o grupo é o fruto (um mesmo fruto muda de classe ao longo dos dias, então a divisão é feita sobre os frutos e escolhe, entre várias tentativas, a que deixa a proporção de classes mais parecida nas três partições).

Nos conjuntos públicos o nome dos arquivos não identifica o fruto (`healthy (N).jpg`, `unripe (N).jpg`), então a divisão é por imagem, depois de remover as cópias. Fotos diferentes do mesmo fruto podem cair em partições diferentes, o que pode inflar um pouco as métricas no teste público; os números do capítulo 5 que valem para R01/R02 saem do teste próprio, dividido por fruto.

Mputu e Khatun são divididos **separadamente** (cada um 70/15/15); treino e validação são juntados e os testes ficam separados, para mostrar o desempenho em cada fonte.
""")

code(r"""
def dividir_por_grupo(itens, grupo_de, fr=(0.70, 0.15, 0.15), tentativas=300, seed=SEED):
    por_grupo = {}
    for f, r in itens:
        por_grupo.setdefault(grupo_de(f), []).append((f, r))
    grupos = sorted(por_grupo)
    n = len(grupos)
    n_tr, n_va = round(n * fr[0]), max(1, round(n * fr[1]))
    alvo = np.mean([r for _, r in itens])
    rng = random.Random(seed)
    melhor = None
    for _ in range(tentativas):
        g = grupos[:]
        rng.shuffle(g)
        sel = {'treino': g[:n_tr], 'val': g[n_tr:n_tr + n_va], 'teste': g[n_tr + n_va:]}
        part = {k: [it for gg in v for it in por_grupo[gg]] for k, v in sel.items()}
        if any(len({r for _, r in v}) < 2 for v in part.values()):
            continue
        custo = sum(abs(np.mean([r for _, r in v]) - alvo) for v in part.values())
        if melhor is None or custo < melhor[0]:
            melhor = (custo, part, sel)
    if melhor is None:
        raise ValueError('Nenhuma divisão deixou as duas classes em todas as partições.')
    return melhor[1], melhor[2]

def salvar_divisao(part, nome):
    with open(SAIDA / f'divisao_{nome}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['particao', 'arquivo', 'classe'])
        for k, v in part.items():
            for arq, r in v:
                w.writerow([k, str(arq), CLASSES[r]])

def resumo_divisao(part, nome):
    info = {}
    for k, v in part.items():
        c = np.bincount([r for _, r in v], minlength=2)
        info[k] = {'total': len(v), 'verde': int(c[0]), 'maduro': int(c[1])}
        print(f'{nome:7s} {k:6s}: {len(v):5d}  (verde {c[0]}, maduro {c[1]})')
    return info

por_imagem = lambda f: str(f)
PART_MPUTU, _ = dividir_por_grupo(ITENS_MPUTU, por_imagem, tentativas=20)
PART_KHATUN, _ = dividir_por_grupo(ITENS_KHATUN, por_imagem, tentativas=20)
PART_PUB = {k: PART_MPUTU[k] + PART_KHATUN[k] for k in ('treino', 'val', 'teste')}
TESTES_PUB = {'mputu': PART_MPUTU['teste'], 'khatun': PART_KHATUN['teste'], 'publico': PART_PUB['teste']}
salvar_divisao(PART_MPUTU, 'mputu')
salvar_divisao(PART_KHATUN, 'khatun')
registrar('divisao_mputu', resumo_divisao(PART_MPUTU, 'Mputu'))
registrar('divisao_khatun', resumo_divisao(PART_KHATUN, 'Khatun'))
registrar('divisao_publico', resumo_divisao(PART_PUB, 'Juntos'))
""")

md(r"""
## 4. Pipeline de imagens, aumento de dados e modelo

Aumento só no treino (seção 3.3): rotação até 20°, espelhamento horizontal, brilho ±20% e zoom leve.
""")

code(r"""
def carregar(caminho, tam):
    b = tf.io.read_file(caminho)
    img = tf.io.decode_image(b, channels=3, expand_animations=False)
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    l = tf.minimum(h, w)
    img = tf.image.crop_to_bounding_box(img, (h - l) // 2, (w - l) // 2, l, l)   # quadrado central
    img = tf.image.resize(img, (tam, tam), method='bilinear')
    return img   # float32, 0..255

def aumentador():
    return keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(20 / 360, fill_mode='reflect'),
        layers.RandomZoom((-0.1, 0.1), fill_mode='reflect'),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name='aumento')

def criar_ds(itens, tam, treino=False):
    caminhos = [str(f) for f, _ in itens]
    rotulos = [r for _, r in itens]
    ds = tf.data.Dataset.from_tensor_slices((caminhos, rotulos))
    if treino:
        ds = ds.shuffle(len(itens), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda c, r: (carregar(c, tam), r), num_parallel_calls=AUTOTUNE)
    if not treino:
        ds = ds.cache()
    ds = ds.batch(LOTE)
    if treino:
        aug = aumentador()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

def criar_modelo(alpha, tam):
    base = keras.applications.MobileNetV2(input_shape=(tam, tam, 3), alpha=alpha,
                                          include_top=False, weights='imagenet')
    base.trainable = False
    ent = keras.Input((tam, tam, 3), name='imagem')                 # pixels 0..255
    x = layers.Rescaling(1 / 127.5, offset=-1, name='normaliza')(ent)  # -> [-1, 1]
    x = base(x, training=False)    # BatchNorm sempre em modo inferência, inclusive no ajuste fino
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.2, name='dropout')(x)
    sai = layers.Dense(2, activation='softmax', name='classe')(x)
    return keras.Model(ent, sai, name=f'mnv2_a{alpha}_{tam}')

def base_de(modelo):
    return next(l for l in modelo.layers if isinstance(l, keras.Model))

def pesos_classe(itens, limite=1.2):
    # inverso da frequência, só se o desbalanceamento passar do limite (seção 3.4)
    c = np.bincount([r for _, r in itens], minlength=2)
    if c.max() / max(c.min(), 1) <= limite:
        return None
    return {i: len(itens) / (2 * c[i]) for i in range(2)}

def treinar(modelo, ds_tr, ds_va, lr, epocas, arquivo, pesos=None):
    modelo.compile(optimizer=keras.optimizers.Adam(lr),
                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
          keras.callbacks.ModelCheckpoint(str(arquivo), monitor='val_loss', save_best_only=True)]
    t0 = time.time()
    h = modelo.fit(ds_tr, validation_data=ds_va, epochs=epocas, callbacks=cb,
                   class_weight=pesos, verbose=2)
    melhor = int(np.argmin(h.history['val_loss']))
    return {'epocas_rodadas': len(h.history['loss']), 'melhor_epoca': melhor + 1,
            'val_loss': float(h.history['val_loss'][melhor]),
            'val_acc': float(h.history['val_accuracy'][melhor]),
            'tempo_s': round(time.time() - t0, 1), 'pesos_classe': pesos}

def metricas(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        'n': int(len(y_true)),
        'acuracia': float(accuracy_score(y_true, y_pred)),
        'precisao_maduro': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'revocacao_maduro': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'f1_maduro': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'matriz_confusao': {'linhas_real': CLASSES, 'colunas_previsto': CLASSES, 'valores': cm.tolist()},
    }

def mostrar(nome, m):
    cm = m['matriz_confusao']['valores']
    print(f"{nome}: n={m['n']}  acurácia={m['acuracia']:.3f}  precisão(maduro)={m['precisao_maduro']:.3f}  "
          f"revocação(maduro)={m['revocacao_maduro']:.3f}  F1={m['f1_maduro']:.3f}")
    print(f"           previsto verde  previsto maduro\n  real verde   {cm[0][0]:6d}  {cm[0][1]:6d}\n"
          f"  real maduro  {cm[1][0]:6d}  {cm[1][1]:6d}")

def avaliar_keras(modelo, itens, tam, nome):
    ds = criar_ds(itens, tam)
    y_pred = modelo.predict(ds, verbose=0).argmax(1)
    m = metricas([r for _, r in itens], y_pred)
    mostrar(nome, m)
    return m
""")

md(r"""
## 5. Etapa 1 – pré-treino (Mputu + Khatun)

A perda é ponderada pelo inverso da frequência das classes, já que juntos os dois conjuntos têm mais maduros que verdes.
""")

code(r"""
for nome in TREINAR:
    cfg = MODELOS[nome]
    tam = cfg['tam']
    print(f'\n######## {nome}: alpha={cfg["alpha"]}, {tam}x{tam} ########')
    modelo = criar_modelo(cfg['alpha'], tam)
    print(f'Parâmetros: {modelo.count_params():,}')
    info = treinar(modelo,
                   criar_ds(PART_PUB['treino'], tam, treino=True),
                   criar_ds(PART_PUB['val'], tam),
                   lr=1e-3, epocas=20, arquivo=SAIDA / f'{nome}_etapa1.keras',
                   pesos=pesos_classe(PART_PUB['treino']))
    info['parametros'] = int(modelo.count_params())
    registrar(f'{nome}/etapa1/treino', info)
    for fonte, itens in TESTES_PUB.items():
        registrar(f'{nome}/etapa1/teste_{fonte}', avaliar_keras(modelo, itens, tam, f'Teste {fonte}'))
""")

md(r"""
## 6. Etapa 2 – ajuste fino com o conjunto próprio

Lê `MyDrive/TCC/dataset/metadata.csv` (gerado pelo `capturar.py`). Imagens apagadas na limpeza (ambíguas, frutos com defeito) são ignoradas e contadas.
Se a pasta ainda não existir, esta seção só avisa e o notebook segue para a quantização com o modelo da etapa 1, o que já permite testar o firmware e medir o tempo de inferência no ESP32.
""")

code(r"""
DESCONGELAR_A_PARTIR = 'block_13_expand'   # "últimos blocos": 13 a 16 + conv final

def ler_proprio():
    meta = PROPRIO / 'metadata.csv'
    if not meta.exists():
        return None
    itens, frutos, faltando = [], {}, 0
    with open(meta, encoding='utf-8') as f:
        for lin in csv.DictReader(f):
            arq = PROPRIO / lin['arquivo'].replace('\\', '/')   # o script roda no Windows
            if not arq.exists():
                faltando += 1
                continue
            itens.append((arq, CLASSES.index(lin['classe'])))
            frutos[arq] = lin['fruto']
    print(f'Conjunto próprio: {len(itens)} imagens, {len(set(frutos.values()))} frutos, '
          f'{faltando} linhas sem arquivo (descartadas na limpeza)')
    registrar('proprio_descartadas', faltando)
    return itens, frutos

dados = ler_proprio()
PART_PROPRIO = None
if dados is None:
    print('Sem conjunto próprio ainda: etapa 2 pulada.')
else:
    ITENS_PROPRIO, FRUTO_DE = dados
    PART_PROPRIO, frutos_part = dividir_por_grupo(ITENS_PROPRIO, lambda f: FRUTO_DE[f])
    salvar_divisao(PART_PROPRIO, 'proprio')
    registrar('divisao_proprio', resumo_divisao(PART_PROPRIO, 'Próprio'))
    registrar('frutos_por_particao', frutos_part)
    print('Frutos por partição:', frutos_part)

    for nome in TREINAR:
        tam = MODELOS[nome]['tam']
        print(f'\n######## {nome}: ajuste fino ########')
        modelo = keras.models.load_model(SAIDA / f'{nome}_etapa1.keras')
        # generalização para a câmera do ESP32: modelo só com dados públicos, nas fotos próprias
        registrar(f'{nome}/etapa1/teste_proprio',
                  avaliar_keras(modelo, PART_PROPRIO['teste'], tam, 'Teste próprio ANTES do ajuste'))
        base = base_de(modelo)
        base.trainable = True
        congelar = True
        for camada in base.layers:
            if camada.name.startswith(DESCONGELAR_A_PARTIR):
                congelar = False
            camada.trainable = (not congelar) and not isinstance(camada, layers.BatchNormalization)
        print('Camadas treináveis na base:', sum(c.trainable for c in base.layers))
        info = treinar(modelo,
                       criar_ds(PART_PROPRIO['treino'], tam, treino=True),
                       criar_ds(PART_PROPRIO['val'], tam),
                       lr=1e-5, epocas=30, arquivo=SAIDA / f'{nome}_etapa2.keras',
                       pesos=pesos_classe(PART_PROPRIO['treino']))
        registrar(f'{nome}/etapa2/treino', info)
        registrar(f'{nome}/etapa2/teste_proprio',
                  avaliar_keras(modelo, PART_PROPRIO['teste'], tam, 'Teste próprio DEPOIS do ajuste'))
        registrar(f'{nome}/etapa2/teste_publico',
                  avaliar_keras(modelo, PART_PUB['teste'], tam, 'Teste público após ajuste'))
""")

md(r"""
## 7. Quantização int8 do modelo embarcado

Usa o modelo da etapa 2 se existir, senão o da etapa 1. Calibração com ~200 imagens do treino (do conjunto próprio quando houver, que é o domínio de uso).
O `.tflite` é avaliado no PC sobre o mesmo teste do modelo float: a diferença mostra o efeito da quantização (seção 3.4).

A quantização desloca p(maduro) para cima (+0,042 em média na validação da etapa 1, igual com 200, 600 ou 1.275 imagens de calibração). O limiar de decisão fica em **0,5** (`AJUSTAR_LIMIAR = False`), mas a célula calcula na **validação** o limiar que devolveria a precisão em "maduro" do float32 e registra a varredura. Se o int8 ficar abaixo de 95% (R02), troque para `True`: o limiar escolhido vai para o `modelo.h` e o firmware passa a usá-lo.
""")

code(r"""
nome, tam = 'embarcado', MODELOS['embarcado']['tam']
etapa = 'etapa2' if (SAIDA / 'embarcado_etapa2.keras').exists() and PART_PROPRIO else 'etapa1'
modelo = keras.models.load_model(SAIDA / f'{nome}_{etapa}.keras')
treino_calib = PART_PROPRIO['treino'] if etapa == 'etapa2' else PART_PUB['treino']
teste_q = PART_PROPRIO['teste'] if etapa == 'etapa2' else PART_PUB['teste']
print('Quantizando o modelo da', etapa)

amostra = random.Random(SEED).sample(treino_calib, min(200, len(treino_calib)))
def representativo():
    for f, _ in amostra:
        yield [tf.expand_dims(carregar(str(f), tam), 0)]

dir_sm = str(TRAB / 'embarcado_savedmodel')
modelo.export(dir_sm)   # Keras 3: o conversor é mais estável a partir do SavedModel
conv = tf.lite.TFLiteConverter.from_saved_model(dir_sm)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representativo
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tfl = conv.convert()
arq_tfl = SAIDA / f'embarcado_{etapa}_int8.tflite'
arq_tfl.write_bytes(tfl)
print(f'{arq_tfl.name}: {len(tfl)/1024:.1f} KB')

tf.lite.experimental.Analyzer.analyze(model_content=tfl)   # lista de operadores para o MicroMutableOpResolver
""")

code(r"""
# Kernels de referência (sem XNNPACK): dão saída int8 idêntica à do ESP32 com ESP-NN.
# O interpretador padrão usa XNNPACK, que arredonda diferente (diário, entrada 26).
interp = tf.lite.Interpreter(model_content=tfl,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
interp.allocate_tensors()
ent, sai = interp.get_input_details()[0], interp.get_output_details()[0]
esc_in, zp_in = ent['quantization']
esc_out, zp_out = sai['quantization']
print(f'Entrada int8: escala={esc_in:.6f}, ponto zero={zp_in}   ->  q = round(pixel/escala) + ponto_zero')
print(f'Saída   int8: escala={esc_out:.6f}, ponto zero={zp_out}')

def prob_maduro_tflite(itens):
    p = []
    for f, _ in itens:
        x = carregar(str(f), tam).numpy()
        q = np.clip(np.round(x / esc_in) + zp_in, -128, 127).astype(np.int8)[None]
        interp.set_tensor(ent['index'], q)
        interp.invoke()
        p.append((float(interp.get_tensor(sai['index'])[0][1]) - zp_out) * esc_out)
    return np.array(p)

def prob_maduro_keras(itens):
    return modelo.predict(criar_ds(itens, tam), verbose=0)[:, 1]

# Limiar de decisão do int8. Por decisão do autor fica em 0,5 (AJUSTAR_LIMIAR = False).
# A varredura na VALIDAÇÃO é sempre calculada e registrada: o limiar sugerido é o menor em que
# a precisão em "maduro" do int8 volta ao nível do float32. A quantização empurra p(maduro)
# para cima (diário, entrada 20). Se o int8 ficar abaixo de 95% (R02), ligar AJUSTAR_LIMIAR.
AJUSTAR_LIMIAR = False
val_q = PART_PROPRIO['val'] if etapa == 'etapa2' else PART_PUB['val']
y_val_q = np.array([r for _, r in val_q])
prec_float_val = metricas(y_val_q, (prob_maduro_keras(val_q) >= 0.5).astype(int))['precisao_maduro']
p_val_int8 = prob_maduro_tflite(val_q)
varredura, LIMIAR = [], None
for t in np.round(np.arange(0.50, 0.96, 0.05), 2):
    m = metricas(y_val_q, (p_val_int8 >= t).astype(int))
    varredura.append({'limiar': float(t), **{k: m[k] for k in ('acuracia', 'precisao_maduro', 'revocacao_maduro')},
                      'matriz': m['matriz_confusao']['valores']})
    if LIMIAR is None and m['precisao_maduro'] >= prec_float_val:
        LIMIAR = float(t)
    print(f"validação int8, limiar {t:.2f}: acc={m['acuracia']:.3f} prec={m['precisao_maduro']:.3f} "
          f"rev={m['revocacao_maduro']:.3f} matriz={m['matriz_confusao']['valores']}")
LIMIAR_SUGERIDO = LIMIAR if LIMIAR is not None else 0.5
print(f'precisão(maduro) do float32 na validação: {prec_float_val:.3f} -> limiar sugerido para o int8: {LIMIAR_SUGERIDO}')
LIMIAR = LIMIAR_SUGERIDO if AJUSTAR_LIMIAR else 0.5
print(f'Limiar usado (AJUSTAR_LIMIAR={AJUSTAR_LIMIAR}): {LIMIAR}')

y_teste_q = [r for _, r in teste_q]
m_float = avaliar_keras(modelo, teste_q, tam, f'Teste ({etapa}) float32, limiar 0,5')
p_teste_int8 = prob_maduro_tflite(teste_q)
m_int8 = metricas(y_teste_q, (p_teste_int8 >= 0.5).astype(int))
mostrar(f'Teste ({etapa}) int8, limiar 0,5', m_int8)
m_int8_lim = metricas(y_teste_q, (p_teste_int8 >= LIMIAR).astype(int))
mostrar(f'Teste ({etapa}) int8, limiar {LIMIAR}', m_int8_lim)
registrar(f'embarcado/{etapa}/quantizacao', {
    'tamanho_kb': round(len(tfl) / 1024, 1),
    'entrada': {'escala': float(esc_in), 'ponto_zero': int(zp_in)},
    'saida': {'escala': float(esc_out), 'ponto_zero': int(zp_out)},
    'float32': m_float, 'int8_limiar_0.5': m_int8,
    'limiar_int8': LIMIAR, 'limiar_sugerido': LIMIAR_SUGERIDO, 'ajustar_limiar': AJUSTAR_LIMIAR,
    'varredura_validacao': varredura,
    'precisao_float_validacao': prec_float_val,
    'int8_limiar_escolhido': m_int8_lim,
    'queda_acuracia_pp': round(100 * (m_float['acuracia'] - m_int8_lim['acuracia']), 2),
})
""")

md(r"""
## 8. Exportação para o firmware

Gera `modelo.h` e `modelo.cpp` com o vetor `g_modelo` (alinhado em 16 bytes). Copie os dois para a pasta do sketch.
""")

code(r"""
def para_c(dados, pasta, nome='g_modelo'):
    linhas = [', '.join(f'0x{b:02x}' for b in dados[i:i + 12]) for i in range(0, len(dados), 12)]
    h = ('#pragma once\n'
         f'// Gerado pelo notebook de treino. Entrada {tam}x{tam}x3 int8: '
         f'escala {esc_in:.6f}, ponto zero {zp_in}.\n'
         f'#define MODELO_ENTRADA_LADO {tam}\n'
         f'#define MODELO_ENTRADA_ZP {zp_in}\n'
         f'#define MODELO_LIMIAR_MADURO {LIMIAR:.2f}f   // escolhido na validação (seção 7)\n'
         f'extern const unsigned char {nome}[];\n'
         f'extern const unsigned int {nome}_len;\n')
    cpp = ('#include "modelo.h"\n\n'
           f'alignas(16) const unsigned char {nome}[] = {{\n  ' + ',\n  '.join(linhas) + '\n};\n'
           f'const unsigned int {nome}_len = {len(dados)};\n')
    (pasta / 'modelo.h').write_text(h)
    (pasta / 'modelo.cpp').write_text(cpp)
    print('Gravados:', pasta / 'modelo.h', pasta / 'modelo.cpp')

para_c(tfl, SAIDA)
""")

md(r"""
## 9. Modelo do servidor em TFLite float32

Mesmo modelo α = 1,0 / 224×224, **sem quantização** (seção 3.4), só trocando o formato: o `.tflite` roda no Orange Pi com o `ai-edge-litert`, sem instalar o TensorFlow inteiro. A célula confere que as saídas batem com as do `.keras`.
""")

code(r"""
if 'servidor' in TREINAR or (SAIDA / 'servidor_etapa1.keras').exists():
    t_srv = MODELOS['servidor']['tam']
    et_srv = 'etapa2' if (SAIDA / 'servidor_etapa2.keras').exists() and PART_PROPRIO else 'etapa1'
    m_srv = keras.models.load_model(SAIDA / f'servidor_{et_srv}.keras')
    dir_srv = str(TRAB / 'servidor_savedmodel')
    m_srv.export(dir_srv)
    tfl_srv = tf.lite.TFLiteConverter.from_saved_model(dir_srv).convert()
    arq_srv = SAIDA / f'servidor_{et_srv}_float.tflite'
    arq_srv.write_bytes(tfl_srv)
    it = tf.lite.Interpreter(model_content=tfl_srv); it.allocate_tensors()
    e, s = it.get_input_details()[0], it.get_output_details()[0]
    dif = 0.0
    for f, _ in PART_PUB['teste'][:20]:
        x = carregar(str(f), t_srv).numpy()[None]
        it.set_tensor(e['index'], x); it.invoke()
        dif = max(dif, float(np.abs(it.get_tensor(s['index']) - m_srv(x, training=False).numpy()).max()))
    print(f'{arq_srv.name}: {len(tfl_srv)/1024/1024:.1f} MB; maior diferença para o .keras: {dif:.2e}')

print(json.dumps(RESULTADOS, indent=1, ensure_ascii=False)[:3000])
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
# compila cada célula de código para pegar erro de sintaxe
for c in cells:
    if c["cell_type"] == "code":
        compile(c["source"], "<cell>", "exec")
for c in cells:  # nbformat quer lista de linhas
    c["source"] = c["source"].splitlines(keepends=True)
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(nb, indent=1, ensure_ascii=False))
print("ok", len(cells), "células")
