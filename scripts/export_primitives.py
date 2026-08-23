"""
Exporta o oraculo das primitivas numericas e textuais.

Diferente do golden.json (que cobre o pipeline de ponta a ponta), este arquivo fixa o
comportamento das funcoes de base — arredondamento, title case, indices cientometricos e
TF-IDF. Sao elas que, se divergirem em silencio, contaminam todas as tabelas.

Uso:
    .venv/bin/python scripts/export_primitives.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "web" / "tests" / "parity" / "primitives.json"

ANO_BASE = 2026


def casos_round() -> list[dict]:
    """Casos de arredondamento, com enfase nos empates exatos.

    Racionais diadicos (k/8, k/16) caem exatamente no meio do digito seguinte, que e onde
    o round() do Python (ties-to-even) diverge do toFixed() do JavaScript (ties-away).
    """
    random.seed(7)
    casos: list[tuple[float, int]] = []

    for k in range(1, 65):
        for den in (2, 4, 8, 16, 32):
            for d in (0, 1, 2, 3):
                casos.append((k / den, d))

    for v in [2.675, 1.005, 0.125, 0.375, 2.5, 3.5, -2.5, 0.615, 1.115, 2.345, 8.835]:
        for d in (0, 1, 2, 3):
            casos.append((v, d))

    # Medias de citacoes realistas: total inteiro dividido por contagem de documentos.
    for _ in range(400):
        casos.append((random.randint(0, 5000) / random.randint(1, 64), 2))

    for _ in range(200):
        casos.append((random.uniform(-1000, 1000), random.choice([0, 1, 2, 3, 4])))

    return [{"value": v, "digits": d, "expected": round(v, d)} for v, d in casos]


def casos_title() -> list[dict]:
    """str.title() do Python: digitos e pontuacao encerram a palavra."""
    entradas = [
        "silva, a.b.",
        "o'brien",
        "van der berg",
        "MCDONALD",
        "abc123def",
        "x1y2z3",
        "  espacos   multiplos  ",
        "ANÁLISE BIBLIOMÉTRICA",
        "josé da silva-santos",
        "3m corporation",
        "e-learning",
        "covid-19 pandemic",
        "",
        "a",
        "ÑOÑO",
        "İstanbul",
    ]
    return [{"input": s, "expected": s.title()} for s in entradas]


def casos_indices() -> list[dict]:
    """Indices h, g, i10 e m sobre distribuicoes de citacao variadas."""
    import sys
    from unittest import mock

    fake_st = mock.MagicMock()
    fake_st.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda fn: fn))
    sys.modules["streamlit"] = fake_st
    for sub in ("streamlit.components", "streamlit.components.v1"):
        sys.modules[sub] = mock.MagicMock()
    sys.path.insert(0, str(RAIZ))

    import pandas as pd

    import utils

    utils.CURRENT_YEAR = ANO_BASE

    random.seed(11)
    cenarios: list[tuple[list[int], list[int] | None]] = [
        ([], None),
        ([0], [2020]),
        ([0, 0, 0], [2020, 2021, 2022]),
        ([1, 1, 1, 1], [2019]),
        ([10, 8, 5, 4, 3], [2015, 2016, 2017, 2018, 2019]),
        ([100, 50, 20, 10, 5, 1], [2010, 2012, 2014, 2016, 2018, 2020]),
        ([9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [2021] * 10),
        # Caso onde g extrapola h de forma acentuada.
        ([100, 1, 1, 1, 1], [2018] * 5),
        # Ano-base igual ao primeiro ano: divisor 1.
        ([50, 40, 30], [ANO_BASE] * 3),
        # Ano futuro: divisor negativo, m deve permanecer 0.
        ([50, 40, 30], [ANO_BASE + 5] * 3),
    ]

    for _ in range(60):
        n = random.randint(1, 40)
        cits = [random.randint(0, 300) for _ in range(n)]
        anos = [random.randint(1995, ANO_BASE) for _ in range(n)]
        cenarios.append((cits, anos))

    casos = []
    for cits, anos in cenarios:
        serie = pd.Series(cits, dtype="float64")
        serie_anos = pd.Series(anos, dtype="float64") if anos else None
        h, g, i10, m = utils.extrair_indices_cientometricos(serie, serie_anos, ano_base=ANO_BASE)
        casos.append(
            {
                "citations": cits,
                "years": anos,
                "expected": {"h": int(h), "g": int(g), "i10": int(i10), "m": float(m)},
            }
        )
    return casos


def caso_tfidf() -> dict:
    """TF-IDF e pares de cosseno sobre um corpus pequeno mas realista."""
    corpus = [
        "machine learning for health care systems",
        "machine learning in health care",
        "deep neural networks for image recognition",
        "convolutional neural networks image classification",
        "bibliometric analysis of scientific production",
        "scientometric analysis of research output",
        "memetic algorithms for combinatorial optimization",
        "evolutionary algorithms and optimization problems",
        "the role of culture in the diffusion of memes",
        "cultural evolution and meme transmission",
    ]

    saida: dict = {"corpus": corpus, "configs": {}}
    for rotulo, kwargs in [
        ("unigram", dict(stop_words="english")),
        ("bigram", dict(stop_words="english", ngram_range=(1, 2), min_df=1)),
    ]:
        vec = TfidfVectorizer(**kwargs)
        matriz = vec.fit_transform(corpus)
        inverso = {int(i): t for t, i in vec.vocabulary_.items()}

        similaridade = cosine_similarity(matriz, dense_output=False).tocoo()
        pares = sorted(
            [
                [int(r), int(c), float(s)]
                for r, c, s in zip(similaridade.row, similaridade.col, similaridade.data)
                if r < c and s >= 0.15
            ],
            key=lambda p: (p[0], p[1]),
        )

        saida["configs"][rotulo] = {
            "vocabulary": [inverso[i] for i in range(len(inverso))],
            "idf": [float(x) for x in vec.idf_],
            "rows": [
                sorted([[int(i), float(x)] for i, x in zip(matriz[r].indices, matriz[r].data)])
                for r in range(len(corpus))
            ],
            "pairs": pares,
        }
    return saida


def main() -> None:
    dados = {
        "_meta": {"baseYear": ANO_BASE},
        "round": casos_round(),
        "titleCase": casos_title(),
        "indices": casos_indices(),
        "tfidf": caso_tfidf(),
    }

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")

    print(f"primitives.json escrito em {SAIDA}")
    print(f"  round:     {len(dados['round'])} casos")
    print(f"  titleCase: {len(dados['titleCase'])} casos")
    print(f"  indices:   {len(dados['indices'])} casos")
    print(f"  tfidf:     {len(dados['tfidf']['corpus'])} documentos, 2 configuracoes")


if __name__ == "__main__":
    main()
