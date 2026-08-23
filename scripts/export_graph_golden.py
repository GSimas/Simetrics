"""
Exporta o oraculo das metricas de rede, a partir do NetworkX.

Constroi o mesmo grafo heterogeneo de `gerar_tabela_metricas_completas` e roda as
metricas do NetworkX sobre ele.

IMPORTANTE — betweenness: o app chama `nx.betweenness_centrality(G, k=100)` SEM `seed`,
o que sorteia as fontes pelo estado global do random e produz valores diferentes a cada
execucao (no grafo de exemplo, 33% dos nos mudam entre duas rodadas). Esse valor nao e
comparavel. Exportamos o betweenness EXATO (k=None), que e deterministico, e e contra ele
que a implementacao TypeScript e validada.

Uso:
    .venv/bin/python scripts/export_graph_golden.py
"""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "web" / "tests" / "parity" / "graph-golden.json"

ANO_BASE = 2026
ARQUIVOS = {
    "scielo.ris": "SciELO",
    "wos.ris": "Web of Science",
    "scopus.ris": "Scopus",
}


def _carregar_utils():
    fake_st = mock.MagicMock()
    fake_st.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda fn: fn))
    sys.modules["streamlit"] = fake_st
    for sub in ("streamlit.components", "streamlit.components.v1"):
        sys.modules[sub] = mock.MagicMock()

    sys.path.insert(0, str(RAIZ))
    import utils

    utils.CURRENT_YEAR = ANO_BASE
    return utils


def _mock_upload(caminho: Path):
    """Reconstroi o objeto de arquivo que o Streamlit entrega aos processadores.

    O BOM e removido antes de entregar. Com ele, a primeira linha vira "\ufeffTY  - JOUR",
    o rispy nao a reconhece como tag e descarta o primeiro registro do arquivo — o app
    perde 2 documentos da base de exemplo sem avisar. Alinhar a entrada aqui faz a
    comparacao medir os algoritmos em vez de reproduzir o bug dos dois lados.
    """
    conteudo = caminho.read_bytes()
    if conteudo.startswith(b"\xef\xbb\xbf"):
        conteudo = conteudo[3:]
    buffer = io.BytesIO(conteudo)
    buffer.name = caminho.name
    return buffer


def _num(valor):
    """NaN e infinito nao sobrevivem ao JSON estrito; viram None."""
    if valor is None:
        return None
    valor = float(valor)
    return None if math.isnan(valor) or math.isinf(valor) else valor


def main() -> None:
    utils = _carregar_utils()

    import networkx as nx
    import pandas as pd

    df = utils.padronizar_base_bibliometrica(
        utils.process_multiple_ris([_mock_upload(RAIZ / n) for n in ARQUIVOS], dict(ARQUIVOS))
    )

    # Mesma construcao de gerar_tabela_metricas_completas (utils.py:2153).
    nodes, edges, tipos = [], [], {}
    for _, linha in df.iterrows():
        doc = str(linha["TITLE"]) if pd.notna(linha["TITLE"]) else None
        if not doc:
            continue
        nodes.append(doc)
        tipos[doc] = "Documento"

        for autor in [x.strip() for x in str(linha["AUTHORS"]).split(";") if x.strip()]:
            nodes.append(autor)
            tipos[autor] = "Autor"
            edges.append((doc, autor))

        for pais in [x.strip() for x in str(linha["COUNTRY"]).split(";") if x.strip()]:
            nodes.append(pais)
            tipos[pais] = "País"
            edges.append((doc, pais))

        venue = str(linha["SECONDARY TITLE"]).strip()
        if venue:
            nodes.append(venue)
            tipos[venue] = "Local de Publicação (Venue)"
            edges.append((doc, venue))

    G = nx.Graph()
    G.add_nodes_from(set(nodes))
    G.add_edges_from(set(edges))
    print(f"grafo: {G.number_of_nodes()} nos, {G.number_of_edges()} arestas", file=sys.stderr)

    # eigenvector_centrality_numpy LEVANTA AmbiguousSolution em grafo desconexo — o
    # NetworkX se recusa a calcular. O app engole isso com um `except` e preenche a coluna
    # inteira com zero (utils.py:2038), entao a centralidade de autovetor da tabela SNA
    # simplesmente nao existe hoje.
    #
    # Para validar a implementacao TypeScript de verdade, calculamos o autovetor no MAIOR
    # COMPONENTE CONEXO, onde o NetworkX funciona normalmente.
    maior = max(nx.connected_components(G), key=len)
    sub = G.subgraph(maior).copy()
    eigen_componente = nx.eigenvector_centrality_numpy(sub)
    print(f"maior componente: {len(maior)} nos", file=sys.stderr)

    try:
        nx.eigenvector_centrality_numpy(G)
        eigen_grafo_todo_falha = False
    except Exception as exc:
        eigen_grafo_todo_falha = True
        print(f"eigenvector no grafo todo: {type(exc).__name__}", file=sys.stderr)

    grau = dict(G.degree())
    grau_cent = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)  # EXATO, deterministico
    pr = nx.pagerank(G)

    graus = [d for _, d in G.degree()]
    import numpy as np

    contagens = np.bincount(graus)
    probs = contagens[contagens > 0] / G.number_of_nodes()
    entropia = float(-np.sum(probs * np.log2(probs)))

    y = contagens[contagens > 0]
    x = np.nonzero(contagens)[0]
    if len(x) > 2:
        inclinacao, _ = np.polyfit(np.log10(x), np.log10(y), 1)
        lei_potencia = abs(float(inclinacao))
    else:
        lei_potencia = 0.0

    s_grau = pd.Series([G.degree(n) for n in G.nodes()])
    s_betw = pd.Series([betweenness[n] for n in G.nodes()])

    golden = {
        "_meta": {
            "arquivos": ARQUIVOS,
            "baseYear": ANO_BASE,
            "eigenvectorFalhaNoGrafoTodo": eigen_grafo_todo_falha,
        },
        # Autovetor no maior componente conexo, normalizado em L2 pelo NetworkX.
        "eigenvectorMaiorComponente": {
            "nodeCount": len(maior),
            "values": {n: _num(v) for n, v in eigen_componente.items()},
        },
        "estrutura": {
            "nodeCount": G.number_of_nodes(),
            "edgeCount": G.number_of_edges(),
            "componentCount": nx.number_connected_components(G),
        },
        "global": {
            "density": _num(nx.density(G)),
            "clustering": _num(nx.average_clustering(G)),
            "entropy": _num(entropia),
            "efficiency": _num(nx.global_efficiency(G)) if len(G) < 1500 else None,
            "meanDegree": _num(np.mean(graus)),
            "stdDegree": _num(np.std(graus)),
            "minDegree": _num(np.min(graus)),
            "maxDegree": _num(np.max(graus)),
            "meanPageRank": _num(np.mean(list(pr.values()))),
                "powerLawExponent": _num(lei_potencia),
            "assortativity": _num(nx.degree_assortativity_coefficient(G)),
            "spearmanDegreeBetweenness": _num(s_grau.corr(s_betw, method="spearman")),
        },
        # Ordenado por grau decrescente, com desempate pelo nome para ser estavel.
        "nodes": [
            {
                "item": n,
                "kind": tipos.get(n, "Outro"),
                "degreeAbsolute": int(grau[n]),
                "degreeCentrality": _num(grau_cent[n]),
                "betweenness": _num(betweenness[n]),
                "closeness": _num(closeness[n]),
            }
            for n in sorted(G.nodes(), key=lambda n: (-grau[n], n))
        ],
    }

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(golden, ensure_ascii=False), encoding="utf-8")

    print(f"graph-golden.json escrito em {SAIDA}", file=sys.stderr)
    print(f"  nos exportados: {len(golden['nodes'])}", file=sys.stderr)
    print(f"  densidade={golden['global']['density']:.6f}", file=sys.stderr)
    print(f"  clustering={golden['global']['clustering']:.6f}", file=sys.stderr)
    print(f"  assortatividade={golden['global']['assortativity']:.6f}", file=sys.stderr)
    zeros = sum(1 for v in betweenness.values() if v == 0.0)
    print(f"  betweenness exatamente zero: {zeros}/{len(betweenness)}", file=sys.stderr)


if __name__ == "__main__":
    main()
