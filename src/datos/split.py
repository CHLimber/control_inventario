"""Grupos anti-fuga, asignación de splits y verificación de fugas.

Un GRUPO reúne todas las imágenes que salen de la misma foto original:
  - mismo proyecto de Roboflow + mismo id original (las copias aumentadas <foto>.rf.<hash> y la
    misma foto exportada en otra versión del proyecto), y
  - imágenes con pHash idéntico (duplicados, incluso entre datasets distintos).
El split se decide por grupo, así ninguna foto aparece en dos splits.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from src.datos.util import SPLITS

PROPORCIONES = (0.8, 0.1, 0.1)
# Tope de imágenes por pHash para unirlas. Sin tope en la práctica: darians_bottle tiene ráfagas de
# hasta ~100 fotos de la misma escena con pHash idéntico, que SÍ deben quedar en el mismo split.
# Las imágenes degeneradas (pHash todo ceros/unos) se excluyen aparte.
MAX_BUCKET_PHASH = 1_000_000
# Un id original con más imágenes que esto no identifica una foto: se subdivide por pHash cercano.
MAX_POR_ID = 20
UMBRAL_HAMMING = 10
_PHASH_VACIO = {"", "0000000000000000", "ffffffffffffffff", "8000000000000000"}


class _UnionFind:
    def __init__(self) -> None:
        self.padre: dict[str, str] = {}

    def buscar(self, x: str) -> str:
        self.padre.setdefault(x, x)
        while self.padre[x] != x:
            self.padre[x] = self.padre[self.padre[x]]
            x = self.padre[x]
        return x

    def unir(self, a: str, b: str) -> None:
        ra, rb = self.buscar(a), self.buscar(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.padre[rb] = ra


def _bits(phash: str) -> np.ndarray:
    return np.array([(int(c, 16) >> k) & 1 for c in phash for k in (3, 2, 1, 0)], dtype=np.uint8)


def _subgrupos_por_phash(phashes: list[str], umbral: int) -> list[int]:
    """Componentes conexas uniendo imágenes con distancia de Hamming (pHash) <= umbral."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    h = np.stack([_bits(p) for p in phashes])
    dist = (h[:, None, :] != h[None, :, :]).sum(-1)
    _, etiquetas = connected_components(csr_matrix(dist <= umbral), directed=False)
    return etiquetas.tolist()


def asignar_grupos(df: pd.DataFrame, max_bucket: int = MAX_BUCKET_PHASH, max_por_id: int = MAX_POR_ID,
                   umbral_hamming: int = UMBRAL_HAMMING) -> pd.DataFrame:
    """Agrega la columna 'grupo' (id estable del grupo anti-fuga).

    Si un mismo id original tiene más de `max_por_id` imágenes, el nombre no identifica una foto
    (p. ej. lamar_grocery sube cientos de fotos llamadas '01', '02'...): ese id se subdivide en
    componentes de pHash cercano (Hamming <= umbral), que reúnen las copias aumentadas de cada foto
    y también fotos casi idénticas de la misma escena.
    """
    df = df.copy()
    claves = (df["proyecto"].astype(str) + "::" + df["id_original"].astype(str)).tolist()
    tamanos = pd.Series(claves).value_counts()
    grandes = set(tamanos[tamanos > max_por_id].index)
    if grandes:
        phashes = df["phash"].astype(str).tolist()
        indices: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(claves):
            if c in grandes:
                indices[c].append(i)
        for c, idx in indices.items():
            for i, sub in zip(idx, _subgrupos_por_phash([phashes[i] for i in idx], umbral_hamming)):
                claves[i] = f"{c}#{sub}"
    uf = _UnionFind()
    for c in claves:
        uf.buscar(c)
    buckets: dict[str, list[str]] = defaultdict(list)
    for c, ph in zip(claves, df["phash"].astype(str)):
        if ph not in _PHASH_VACIO:
            buckets[ph].append(c)
    for miembros in buckets.values():
        distintos = sorted(set(miembros))
        if 1 < len(distintos) <= max_bucket:
            for m in distintos[1:]:
                uf.unir(distintos[0], m)
    df["foto"] = claves  # una foto original (sus copias aumentadas comparten 'foto')
    df["grupo"] = [uf.buscar(c) for c in claves]
    return df


def _fraccion(grupo: str, semilla: int) -> float:
    h = hashlib.sha1(f"{semilla}:{grupo}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def asignar_splits(df: pd.DataFrame, usos: dict[str, str], proporciones=PROPORCIONES,
                   semilla: int = 42) -> pd.DataFrame:
    """Agrega la columna 'split'.

    - Si algún miembro del grupo viene de un dataset con uso 'test_externo', el grupo entero va a test.
    - El resto se reparte por hash determinista del grupo (estable aunque se agreguen datasets).
    """
    df = df.copy()
    externos = set(df.loc[df["alias"].map(lambda a: usos.get(a) == "test_externo"), "grupo"])
    p_train, p_val = proporciones[0], proporciones[0] + proporciones[1]
    splits = {}
    for g in df["grupo"].unique():
        if g in externos:
            splits[g] = "test"
            continue
        f = _fraccion(g, semilla)
        splits[g] = "train" if f < p_train else ("val" if f < p_val else "test")
    df["split"] = df["grupo"].map(splits)
    return df


def seleccionar(df: pd.DataFrame, copias_train: int = 1, prioridad_alias: list[str] | None = None) -> pd.DataFrame:
    """Elige qué imágenes usar. La unidad es la 'foto' (una foto original y sus copias aumentadas).

    - train: hasta `copias_train` imágenes por (alias, foto) (0 = todas). YOLO y Keras ya aumentan
      en línea, así que las copias aumentadas offline de Roboflow aportan poco.
    - val/test: UNA imagen por foto (las copias aumentadas inflarían las métricas).
    - Se quitan los duplicados exactos de pHash dentro de cada split (en train, salvo copias_train=0).
    Las fotos distintas de un mismo grupo (p. ej. una ráfaga) se conservan: el grupo ya garantiza
    que todas caen en el mismo split. Se prefieren imágenes que Roboflow no aumentó (venían en
    valid/test) y luego el orden de alias.
    """
    if df.empty:
        return df
    prioridad = {a: i for i, a in enumerate(prioridad_alias or [])}
    d = df.copy()
    d["_p_alias"] = d["alias"].map(lambda a: prioridad.get(a, len(prioridad)))
    d["_p_aug"] = (~d["split_origen"].isin(["valid", "test"])).astype(int)
    d = d.sort_values(["_p_aug", "_p_alias", "ruta"])
    es_train = d["split"] == "train"
    if "foto" not in d.columns:
        d["foto"] = d["grupo"]
    tr = d[es_train]
    if copias_train > 0:
        tr = tr.groupby(["alias", "foto"], sort=False).head(copias_train)
    ev = d[~es_train].groupby(["alias", "foto"], sort=False).head(1)
    sel = pd.concat([tr, ev])
    dup = sel.duplicated(["split", "phash"]) & ~sel["phash"].astype(str).isin(_PHASH_VACIO)
    if copias_train == 0:  # "todas las copias" en train: no se tocan
        dup &= sel["split"] != "train"
    sel = sel[~dup]
    return sel.drop(columns=["_p_alias", "_p_aug"]).sort_values(["split", "alias", "ruta"])


def verificar_fugas(df: pd.DataFrame, col_grupo: str = "grupo", col_phash: str = "phash") -> list[str]:
    """Lista de problemas: grupos o pHash que aparecen en más de un split. Vacía = sin fugas."""
    problemas = []
    if df.empty:
        return problemas
    por_grupo = df.groupby(col_grupo)["split"].nunique()
    for g in por_grupo[por_grupo > 1].index[:20]:
        problemas.append(f"grupo {g!r} en splits {sorted(df.loc[df[col_grupo] == g, 'split'].unique())}")
    if (por_grupo > 1).sum() > 20:
        problemas.append(f"... y {(por_grupo > 1).sum() - 20} grupos más")
    if col_phash in df.columns:
        d = df[~df[col_phash].astype(str).isin(_PHASH_VACIO)]
        # Los buckets degenerados (> MAX_BUCKET_PHASH grupos) no se unen en asignar_grupos: se ignoran.
        tam = d.groupby(col_phash)[col_grupo].nunique()
        d = d[d[col_phash].map(tam) <= MAX_BUCKET_PHASH]
        por_hash = d.groupby(col_phash)["split"].nunique()
        n = int((por_hash > 1).sum())
        if n:
            problemas.append(f"{n} pHash repetidos entre splits distintos")
    desconocidos = set(df["split"].unique()) - set(SPLITS)
    if desconocidos:
        problemas.append(f"splits desconocidos: {desconocidos}")
    return problemas
