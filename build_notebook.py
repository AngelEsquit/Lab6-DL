"""Genera Lab6_Transfer_Learning.ipynb (python build_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Laboratorio 6 - Transfer Learning y Fine-Tuning (CIFAR-10)
CC3092 Deep Learning y Sistemas Inteligentes.

Compara tres modelos: **CNN desde cero**, **VGG-16 como feature extractor** y **VGG-16 con fine-tuning**.

> Cambiar `QUICK = False` para la corrida completa (recomendado en GPU, p. ej. Colab). `QUICK = True` es solo una prueba de humo en CPU con subconjuntos y pocos epochs.
""")

code("""
import os, time, copy, json, random
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from fvcore.nn import FlopCountAnalysis

QUICK = os.environ.get("LAB6_QUICK", "0") == "1"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HW = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
print("QUICK:", QUICK, "| Hardware:", HW, "| torch", torch.__version__)

VGG_RES = 112          # resolución para ambos modelos VGG-16 (justificada en el reporte)
BATCH = 128
NUM_WORKERS = 0 if os.name == "nt" else 2
N_TRAIN_SUB = 2000 if QUICK else None   # submuestreo solo para smoke test
N_VAL_SUB = 500 if QUICK else None
N_TEST_SUB = 500 if QUICK else None
EPOCHS_CNN = 2 if QUICK else 25
EPOCHS_VGG = 1 if QUICK else 8
""")

md("## 2. Exploración y preparación de los datos")
code("""
raw_train = datasets.CIFAR10(root="data", train=True, download=True)
raw_test = datasets.CIFAR10(root="data", train=False, download=True)
classes = raw_train.classes
y_train_all = np.array(raw_train.targets)
print("Train:", len(raw_train), "| Test:", len(raw_test), "| Clases:", len(classes))
print("Train por clase:", np.bincount(y_train_all))
print("Test por clase:", np.bincount(raw_test.targets))
x0, _ = raw_train[0]
print("Resolución/canales:", x0.size, np.array(x0).shape)

X = raw_train.data.astype(np.float32) / 255.0   # (N,32,32,3)
mean_c, std_c = X.mean(axis=(0, 1, 2)), X.std(axis=(0, 1, 2))
print("CIFAR-10 mean:", mean_c.round(4), "std:", std_c.round(4))
print("ImageNet mean: [0.485 0.456 0.406] std: [0.229 0.224 0.225]")
""")
md("""
**Respuestas (completar con la salida):** CIFAR-10 tiene 50,000 imágenes de entrenamiento y 10,000 de test, 10 clases perfectamente balanceadas (5,000 por clase en train, 1,000 en test). Las imágenes son 32x32 RGB (3 canales). Las estadísticas por canal de CIFAR-10 son cercanas a las de ImageNet (media ~0.49/0.48/0.45, std ~0.25/0.24/0.26), por lo que normalizar con ImageNet para VGG-16 es razonable y es lo que espera el modelo preentrenado.
""")
code("""
fig, axes = plt.subplots(10, 5, figsize=(7, 14))
for c in range(10):
    idx = np.where(y_train_all == c)[0][:5]
    for j, i in enumerate(idx):
        axes[c, j].imshow(raw_train.data[i]); axes[c, j].axis("off")
    axes[c, 0].set_title(classes[c], fontsize=8, loc="left")
plt.tight_layout(); plt.savefig("fig_examples.png", dpi=120); plt.show()
""")
md("""
**Pares difíciles esperados:** gato/perro (poses y texturas similares, baja resolución), automóvil/camión (formas rectangulares y vehículos), ciervo/caballo, pájaro/avión (fondo de cielo) y barco/avión. Con 32x32 los detalles finos (orejas, hocico, ruedas) casi desaparecen.

**Preprocesamiento:**
- CNN desde cero: resolución nativa 32x32, `ToTensor` + normalización con media/std de CIFAR-10. Augmentation en train: `RandomCrop(32, padding=4)` y `RandomHorizontalFlip`.
- VGG-16: `Resize(VGG_RES)` (el modelo espera 224x224, se usa 112x112 por restricciones de cómputo, igual para ambos modelos VGG), `ToTensor` + normalización ImageNet. Mismo augmentation.
- Pasar de 32x32 a 224x224 multiplica los píxeles por 49 (1,024 -> 50,176) y el cómputo convolucional crece aproximadamente igual (~49x). A 112x112 son 12x píxeles y ~12x cómputo, y la salida de `features` es 3x3 en vez de 7x7 (el `avgpool` adaptativo la lleva a 7x7).
- Augmentation: flip horizontal y recorte aleatorio con padding, porque los objetos conservan su clase ante esas transformaciones y reducen el sobreajuste. Se aplica solo en train: validación y test deben medir el desempeño sobre datos reales sin alterar, y la augmentation es un regularizador que simula más datos de entrenamiento, no parte de la distribución a evaluar.
""")
code("""
tr_idx, va_idx = train_test_split(np.arange(len(y_train_all)), test_size=5000,
                                  stratify=y_train_all, random_state=SEED)
print(len(tr_idx), len(va_idx), np.bincount(y_train_all[va_idx]))

def stratified_sub(idx, y, n):
    if n is None or n >= len(idx): return idx
    s, _ = train_test_split(idx, train_size=n, stratify=y[idx], random_state=SEED)
    return s
te_idx = stratified_sub(np.arange(len(raw_test)), np.array(raw_test.targets), N_TEST_SUB)
tr_idx = stratified_sub(tr_idx, y_train_all, N_TRAIN_SUB)
va_idx = stratified_sub(va_idx, y_train_all, N_VAL_SUB)

CIFAR_N = transforms.Normalize(mean_c.tolist(), std_c.tolist())
IMNET_N = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]

def make_tf(kind, train):
    t = list(aug) if train else []
    if kind == "vgg": t.append(transforms.Resize(VGG_RES))
    t += [transforms.ToTensor(), CIFAR_N if kind == "cnn" else IMNET_N]
    return transforms.Compose(t)

def loaders(kind, train_idx=None):
    train_idx = tr_idx if train_idx is None else train_idx
    d_tr = datasets.CIFAR10("data", train=True, transform=make_tf(kind, True))
    d_va = datasets.CIFAR10("data", train=True, transform=make_tf(kind, False))
    d_te = datasets.CIFAR10("data", train=False, transform=make_tf(kind, False))
    mk = lambda d, i, sh: DataLoader(Subset(d, i), batch_size=BATCH, shuffle=sh,
                                     num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    return mk(d_tr, train_idx, True), mk(d_va, va_idx, False), mk(d_te, te_idx, False)
""")

md("""
## 3. Investigación: transfer learning en PyTorch

- **`torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)`**: construye VGG-16 (13 conv + 3 FC, ~138M parámetros). `weights` selecciona los pesos preentrenados (`None` = aleatorio, `DEFAULT` = los mejores). `VGG16_Weights.IMAGENET1K_V1.transforms()` devuelve el preprocesamiento esperado: `Resize(256)`, `CenterCrop(224)`, `ToTensor`, normalización ImageNet. Parámetro relevante de `vgg16`: `progress`.
- **Estructura**: `model.features` (`nn.Sequential` de 13 conv+ReLU y 5 `MaxPool2d`, divididos en 5 bloques; produce 512 mapas), `model.avgpool` (`nn.AdaptiveAvgPool2d((7,7))`, fija la salida a 7x7 sin importar la entrada; parámetro `output_size`) y `model.classifier` (Linear 25088->4096, ReLU, Dropout, Linear 4096->4096, ReLU, Dropout, Linear 4096->1000). Se reemplaza `classifier[6]` por `Linear(4096, 10)`.
- **`requires_grad`**: si es `False` el parámetro no acumula gradiente y el optimizador no lo actualiza. Se congela con `for p in model.features.parameters(): p.requires_grad = False`; por bloques, recorriendo `features[a:b]`. Se pasan al optimizador solo los parámetros con `requires_grad=True`.
- **`model.train()` vs `model.eval()`**: `train()` activa Dropout y usa estadísticas del batch en BatchNorm; `eval()` desactiva Dropout y usa estadísticas acumuladas. **`torch.no_grad()`** desactiva el grafo de autograd (menos memoria y más rápido) en validación/inferencia; no sustituye a `eval()`.
- **`param_groups`**: el optimizador acepta una lista de dicts `[{"params": backbone, "lr": 1e-5}, {"params": head, "lr": 1e-3}]` para usar learning rates distintos por grupo (clave en fine-tuning).
- **Conteo de costo**: `torchinfo.summary` (parámetros y forma), `fvcore.nn.FlopCountAnalysis` (cuenta una operación por multiply-accumulate, o sea MACs, que reporta como "flops"), `thop.profile` (MACs) y `ptflops`. Aquí se usa fvcore.
- **Memoria/tiempo en GPU**: `torch.cuda.reset_peak_memory_stats()` + `torch.cuda.max_memory_allocated()` dan la memoria pico; como los kernels CUDA son asíncronos, hay que llamar `torch.cuda.synchronize()` antes de leer el reloj para medir tiempo correcto.
- **MACs vs FLOPs**: 1 MAC = multiplicación + suma = 2 FLOPs. Una conv/Linear con N MACs cuesta ~2N FLOPs.
- **Parámetros totales vs entrenables**: totales = todos los tensores del modelo; entrenables = los que tienen `requires_grad=True` y se actualizan. Congelar reduce parámetros entrenables, memoria de gradientes/optimizador y cómputo del backward, pero no el forward.
""")

md("## 4. Modelos y utilidades de entrenamiento")
code("""
class CNN(nn.Module):
    def __init__(self, widths=(32, 64, 128), p=0.3):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers += [nn.Conv2d(cin, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(True),
                       nn.MaxPool2d(2), nn.Dropout(p)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(cin * 4 * 4, 256), nn.ReLU(True),
                                        nn.Dropout(0.5), nn.Linear(256, 10))
    def forward(self, x): return self.classifier(self.features(x))

# Bloques de VGG-16: indices de model.features
VGG_BLOCKS = {1: (0, 5), 2: (5, 10), 3: (10, 17), 4: (17, 24), 5: (24, 31)}

def build_vgg(unfreeze_blocks=()):
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    m.classifier[6] = nn.Linear(4096, 10)
    for p in m.features.parameters(): p.requires_grad = False
    for b in unfreeze_blocks:
        a, e = VGG_BLOCKS[b]
        for p in m.features[a:e].parameters(): p.requires_grad = True
    return m

def count_params(m):
    tot = sum(p.numel() for p in m.parameters())
    return tot, sum(p.numel() for p in m.parameters() if p.requires_grad)

def evaluate(model, loader, criterion=None):
    model.eval(); ys, ps, loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if criterion is not None: loss += criterion(out, y).item() * len(y)
            n += len(y); ys.append(y.cpu()); ps.append(out.argmax(1).cpu())
    ys, ps = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    p, r, f, _ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
    return dict(loss=loss / n, acc=accuracy_score(ys, ps), prec=p, rec=r, f1=f), ys, ps

def sync():
    if device.type == "cuda": torch.cuda.synchronize()

RESULTS = []   # una fila por iteracion

def run_experiment(name, family, model, loaders_, epochs, groups, desc, frozen, wd=0.0, label_smooth=0.0):
    \"\"\"groups: lista de dicts param_groups. Guarda el mejor modelo segun F1 de validacion.\"\"\"
    tr, va, _ = loaders_
    model = model.to(device)
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    tot, trn = count_params(model)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    hist = dict(train_loss=[], val_loss=[], val_acc=[], val_f1=[], epoch_time=[], cum_time=[])
    best, best_state, best_ep, t_all = -1, None, 0, 0.0
    for ep in range(1, epochs + 1):
        model.train(); sync(); t0 = time.time(); run, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y); loss.backward(); opt.step()
            run += loss.item() * len(y); n += len(y)
        sched.step(); sync(); dt = time.time() - t0; t_all += dt
        m, _, _ = evaluate(model, va, nn.CrossEntropyLoss())
        for k, v in zip(hist, [run / n, m["loss"], m["acc"], m["f1"], dt, t_all]): hist[k].append(v)
        if m["f1"] > best: best, best_ep, best_m = m["f1"], ep, m; best_state = copy.deepcopy(model.state_dict())
        print(f"[{name}] ep {ep}/{epochs} train {run/n:.4f} val {m['loss']:.4f} acc {m['acc']:.4f} f1 {m['f1']:.4f} ({dt:.0f}s)")
    mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    model.load_state_dict(best_state)
    row = dict(name=name, family=family, config=desc, frozen=frozen, params_total=tot, params_trainable=trn,
               epochs=epochs, best_epoch=best_ep, time_per_epoch=t_all / epochs, total_time=t_all,
               peak_mem_MB=mem, val_acc=best_m["acc"], val_prec=best_m["prec"], val_rec=best_m["rec"],
               val_f1=best_m["f1"], hist=hist, state=best_state)
    RESULTS.append(row)
    return row

def plot_curves(rows, fname):
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 3.6), squeeze=False)
    for ax, r in zip(axes[0], rows):
        e = range(1, len(r["hist"]["train_loss"]) + 1)
        ax.plot(e, r["hist"]["train_loss"], label="train"); ax.plot(e, r["hist"]["val_loss"], label="val")
        ax.set_title(r["name"], fontsize=9); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.show()

def table(rows):
    return pd.DataFrame([{k: v for k, v in r.items() if k not in ("hist", "state")} for r in rows])
""")

md("### 4.1 CNN desde cero (3 iteraciones)")
code("""
L_cnn = loaders("cnn")
cnn_cfgs = [
    ("CNN-1 base", dict(widths=(32, 64, 128), p=0.3), 1e-3, 0.0),
    ("CNN-2 mas ancha", dict(widths=(64, 128, 256), p=0.3), 1e-3, 5e-4),
    ("CNN-3 ancha + mas dropout", dict(widths=(64, 128, 256), p=0.4), 2e-3, 5e-4),
]
for name, kw, lr, wd in cnn_cfgs:
    torch.manual_seed(SEED)
    m = CNN(**kw)
    run_experiment(name, "CNN", m, L_cnn, EPOCHS_CNN, [{"params": m.parameters(), "lr": lr}],
                   f"{kw}, AdamW lr={lr}, wd={wd}, cosine", "ninguna (todo entrenable)", wd=wd)
cnn_rows = [r for r in RESULTS if r["family"] == "CNN"]
plot_curves(cnn_rows, "fig_curves_cnn.png"); table(cnn_rows)
""")

md("### 4.2 VGG-16 como feature extractor (3 iteraciones)")
code("""
L_vgg = loaders("vgg")
fe_cfgs = [("VGG-FE-1 lr=1e-3", 1e-3, 0.0), ("VGG-FE-2 lr=3e-4", 3e-4, 0.0), ("VGG-FE-3 lr=1e-3 wd=1e-2", 1e-3, 1e-2)]
for name, lr, wd in fe_cfgs:
    torch.manual_seed(SEED)
    m = build_vgg(unfreeze_blocks=())
    run_experiment(name, "VGG-FE", m, L_vgg, EPOCHS_VGG, [{"params": m.classifier.parameters(), "lr": lr}],
                   f"AdamW lr={lr}, wd={wd}, res={VGG_RES}", "features 1-5 congelados; solo classifier", wd=wd)
fe_rows = [r for r in RESULTS if r["family"] == "VGG-FE"]
plot_curves(fe_rows, "fig_curves_fe.png"); table(fe_rows)
""")

md("### 4.3 VGG-16 con fine-tuning (3+ iteraciones; varía bloques descongelados y lr del backbone)")
code("""
ft_cfgs = [("VGG-FT-1 bloque5, lr_bb=1e-5", (5,), 1e-5, 1e-3),
           ("VGG-FT-2 bloques4-5, lr_bb=1e-5", (4, 5), 1e-5, 1e-3),
           ("VGG-FT-3 bloques4-5, lr_bb=1e-4", (4, 5), 1e-4, 1e-3),
           ("VGG-FT-4 bloques3-5, lr_bb=1e-5", (3, 4, 5), 1e-5, 1e-3)]
for name, blocks, lr_bb, lr_h in ft_cfgs:
    torch.manual_seed(SEED)
    m = build_vgg(unfreeze_blocks=blocks)
    bb = [p for p in m.features.parameters() if p.requires_grad]
    run_experiment(name, "VGG-FT", m, L_vgg, EPOCHS_VGG,
                   [{"params": bb, "lr": lr_bb}, {"params": m.classifier.parameters(), "lr": lr_h}],
                   f"AdamW lr_backbone={lr_bb}, lr_head={lr_h}, res={VGG_RES}",
                   f"bloques {blocks} descongelados + classifier")
ft_rows = [r for r in RESULTS if r["family"] == "VGG-FT"]
plot_curves(ft_rows, "fig_curves_ft.png"); table(ft_rows)
""")

md("### Tabla de todas las iteraciones")
code("""
df_all = table(RESULTS)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
df_all[["name", "params_total", "params_trainable", "epochs", "best_epoch", "time_per_epoch", "total_time",
        "peak_mem_MB", "val_acc", "val_prec", "val_rec", "val_f1"]].round(4)
""")

md("## Evaluación final en test (una sola vez) y matrices de confusión")
code("""
best_by_family = {f: max([r for r in RESULTS if r["family"] == f], key=lambda r: r["val_f1"])
                  for f in ["CNN", "VGG-FE", "VGG-FT"]}

def rebuild(r):
    if r["family"] == "CNN":
        kw = [c for c in cnn_cfgs if c[0] == r["name"]][0][1]; m = CNN(**kw)
    else:
        m = build_vgg(())
    m.load_state_dict(r["state"]); return m.to(device)

test_rows, cms = [], {}
for fam, r in best_by_family.items():
    m = rebuild(r)
    L = L_cnn if fam == "CNN" else L_vgg
    met, ys, ps = evaluate(m, L[2])
    cms[fam] = confusion_matrix(ys, ps)
    test_rows.append(dict(family=fam, best=r["name"], test_acc=met["acc"], test_prec=met["prec"],
                          test_rec=met["rec"], test_f1=met["f1"]))
df_test = pd.DataFrame(test_rows); display(df_test.round(4))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (fam, cm) in zip(axes, cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax, cbar=False)
    ax.set_title(f"{fam}: {best_by_family[fam]['name']}"); ax.set_xlabel("predicho"); ax.set_ylabel("real")
plt.tight_layout(); plt.savefig("fig_confusion.png", dpi=120); plt.show()
for fam, cm in cms.items():
    off = cm.copy(); np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:3], off.shape))[0]
    print(fam, "mayores confusiones:", [(classes[i], classes[j], int(off[i, j])) for i, j in top])
""")

md("## 4.1 (enunciado) Experimento: 10 % de los datos")
code("""
FRAC = 0.10
small_idx = stratified_sub(tr_idx, y_train_all, int(len(tr_idx) * FRAC))
print("Imagenes de train usadas:", len(small_idx), np.bincount(y_train_all[small_idx]))
small_rows = {}
for fam, r in best_by_family.items():
    torch.manual_seed(SEED)
    if fam == "CNN":
        kw = [c for c in cnn_cfgs if c[0] == r["name"]][0]
        m = CNN(**kw[1]); lr, wd = kw[2], kw[3]
        res = run_experiment(r["name"] + " [10%]", "CNN-10", m, loaders("cnn", small_idx), EPOCHS_CNN,
                             [{"params": m.parameters(), "lr": lr}], "10% datos", "-", wd=wd)
        L = loaders("cnn", small_idx)
    elif fam == "VGG-FE":
        lr = float(r["config"].split("lr=")[1].split(",")[0]); wd = float(r["config"].split("wd=")[1].split(",")[0])
        m = build_vgg(())
        res = run_experiment(r["name"] + " [10%]", "VGG-FE-10", m, loaders("vgg", small_idx), EPOCHS_VGG,
                             [{"params": m.classifier.parameters(), "lr": lr}], "10% datos", "-", wd=wd)
        L = loaders("vgg", small_idx)
    else:
        cfg = [c for c in ft_cfgs if c[0] == r["name"]][0]
        m = build_vgg(cfg[1]); bb = [p for p in m.features.parameters() if p.requires_grad]
        res = run_experiment(r["name"] + " [10%]", "VGG-FT-10", m, loaders("vgg", small_idx), EPOCHS_VGG,
                             [{"params": bb, "lr": cfg[2]}, {"params": m.classifier.parameters(), "lr": cfg[3]}],
                             "10% datos", "-")
        L = loaders("vgg", small_idx)
    mm = rebuild(dict(r, state=res["state"]))
    met, _, _ = evaluate(mm, L[2])
    small_rows[fam] = met
df_small = pd.DataFrame([dict(family=f, acc_100=df_test.set_index("family").loc[f, "test_acc"], f1_100=df_test.set_index("family").loc[f, "test_f1"],
                              acc_10=small_rows[f]["acc"], f1_10=small_rows[f]["f1"]) for f in best_by_family])
df_small["delta_acc"] = df_small.acc_10 - df_small.acc_100
df_small.round(4)
""")

md("## 5. Comparación de rendimiento y recursos")
code("""
def measure_cost(model, res, n_lat=200):
    model = model.to(device).eval()
    x = torch.randn(1, 3, res, res, device=device)
    fa = FlopCountAnalysis(model, x); fa.unsupported_ops_warnings(False); fa.uncalled_modules_warnings(False)
    macs = fa.total()                       # fvcore cuenta MACs
    xb = torch.randn(32, 3, res, res, device=device)
    with torch.no_grad():
        for _ in range(5): model(xb)
        sync(); t0 = time.time()
        for _ in range(n_lat if device.type == "cuda" else 5): model(xb)
        sync(); n = n_lat if device.type == "cuda" else 5
    return macs, (time.time() - t0) / n / 32 * 1000

def trainable_macs(model, res):
    # MACs del forward por capa; el backward cuesta ~2x forward en capas entrenables
    # y ~1x (solo grad de entrada) en capas congeladas que quedan por debajo de una capa entrenable.
    x = torch.randn(1, 3, res, res, device=device)
    fa = FlopCountAnalysis(model.to(device).eval(), x); fa.unsupported_ops_warnings(False)
    by_mod = fa.by_module()
    first_train = None
    for i, (n_, m) in enumerate(model.features.named_children()):
        if any(p.requires_grad for p in m.parameters()): first_train = i; break
    bw = 0
    for n_, m in model.features.named_children():
        c = by_mod[f"features.{n_}"]
        if any(p.requires_grad for p in m.parameters()): bw += 2 * c
        elif first_train is not None and int(n_) > first_train: bw += c
    for n_, m in model.classifier.named_children():
        c = by_mod[f"classifier.{n_}"]
        if any(p.requires_grad for p in m.parameters()): bw += 2 * c
    return bw

rows = []
n_train = len(tr_idx)
for fam, r in best_by_family.items():
    m = rebuild(r); res = 32 if fam == "CNN" else VGG_RES
    macs, lat = measure_cost(m, res)
    if fam == "CNN":
        bw = 2 * macs
    else:
        m_ = build_vgg(()) ; m_.load_state_dict(r["state"])
        cfg = [c for c in ft_cfgs if c[0] == r["name"]]
        if cfg:
            for b in cfg[0][1]:
                a, e = VGG_BLOCKS[b]
                for p in m_.features[a:e].parameters(): p.requires_grad = True
        bw = trainable_macs(m_, res)
    train_flops = 2 * (macs + bw) * n_train * r["best_epoch"] * (r["epochs"] / r["best_epoch"])  # todas las epochs ejecutadas
    t = df_test.set_index("family").loc[fam]
    rows.append(dict(modelo=r["name"], params_total=r["params_total"], params_train=r["params_trainable"],
                     GMACs_inf=macs / 1e9, GFLOPs_inf=2 * macs / 1e9, latencia_ms_img=lat,
                     t_epoch_s=r["time_per_epoch"], epochs_hasta_mejor=r["best_epoch"], t_total_s=r["total_time"],
                     mem_pico_MB=r["peak_mem_MB"], train_PFLOPs=train_flops / 1e15,
                     test_acc=t.test_acc, test_prec=t.test_prec, test_rec=t.test_rec, test_f1=t.test_f1,
                     f1_10pct=small_rows[fam]["f1"]))
df_cmp = pd.DataFrame(rows); df_cmp.round(4).T
""")
md("""
*Estimación de cómputo de entrenamiento:* `FLOPs_train ≈ 2·(MACs_fwd + MACs_bwd) · N_imagenes · N_epochs`. En la CNN y las capas entrenables el backward cuesta ~2x el forward (gradiente respecto a pesos y a entradas). En VGG-FE el backbone no necesita backward (no hay capa entrenable debajo), solo el classifier; en fine-tuning el backward cubre los bloques descongelados y los gradientes de entrada de las capas congeladas que quedan por encima de ellos... (por debajo no hace falta).
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for r in [best_by_family[f] for f in best_by_family]:
    ax[0].plot(np.array(r["hist"]["cum_time"]), r["hist"]["val_acc"], marker="o", label=r["name"])
ax[0].set_xlabel("tiempo de entrenamiento acumulado (s)"); ax[0].set_ylabel("accuracy de validación"); ax[0].legend(fontsize=7)
ax[0].set_title("Validación vs tiempo")
ax[1].scatter(df_cmp.GFLOPs_inf, df_cmp.test_f1, s=80)
for _, rr in df_cmp.iterrows(): ax[1].annotate(rr.modelo, (rr.GFLOPs_inf, rr.test_f1), fontsize=7)
ax[1].set_xscale("log"); ax[1].set_xlabel("GFLOPs de inferencia por imagen (log)"); ax[1].set_ylabel("F1 macro en test")
ax[1].set_title("F1 vs costo")
plt.tight_layout(); plt.savefig("fig_comparison.png", dpi=120); plt.show()
df_cmp.to_csv("results_summary.csv", index=False); df_all.to_csv("results_iterations.csv", index=False)
print("Hardware:", HW)
""")

md("""
## 6. Discusión y análisis
*(Completar tras la corrida completa con `QUICK=False` usando los números de las tablas y figuras.)*

1. **Mejor modelo en test y diferencia / costo adicional:** ...
2. **Mejor relación desempeño/recursos (gráficas de la sección 5):** ...
3. **Utilidad de features preentrenadas a 32x32 y efecto del redimensionamiento:** las features de bajo nivel (bordes, texturas, colores) y parte de las de nivel medio son transferibles; el upsampling no agrega información pero ajusta la escala de los objetos al tamaño de campo receptivo para el que se entrenó la red. Más resolución suele mejorar el accuracy a costa de ~(res/32)² más cómputo y memoria. ...
4. **Fine-tuning (bloques y lr; overfitting / catastrophic forgetting):** ...
5. **Efecto de reducir datos al 10 %:** ...
6. **Matrices de confusión:** ...
7. **Despliegue (a) servidor con GPU, (b) móvil:** para (b) una arquitectura alternativa a VGG-16 sería MobileNetV3 o EfficientNet-B0 (depthwise separable convolutions; ~5M parámetros y órdenes de magnitud menos FLOPs). ...
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "Lab6_Transfer_Learning.ipynb")
print("ok")
