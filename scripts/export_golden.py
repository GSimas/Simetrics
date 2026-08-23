"""
Exporta o oraculo de paridade a partir do pipeline Streamlit original.

Roda as funcoes reais de utils.py sobre os tres arquivos .ris do repositorio e grava
web/tests/parity/golden.json. Os testes Vitest carregam os mesmos .ris pelo pipeline
TypeScript e comparam contra este arquivo.

Uso:
    .venv/bin/python scripts/export_golden.py

O ano-base e fixado em ANO_BASE para que o oraculo nao mude de valor conforme o relogio:
os indices m e a idade media dependem do ano corrente.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "web" / "tests" / "parity" / "golden.json"

# Congelado de proposito: 'CURRENT_YEAR' do utils.py usa date.today().year, e sem fixar
# isso o golden.json passaria a divergir sozinho na virada do ano.
ANO_BASE = 2026

ARQUIVOS = {
    "scielo.ris": "SciELO",
    "wos.ris": "Web of Science",
    "scopus.ris": "Scopus",
}


def _carregar_utils():
    """Importa utils.py neutralizando o Streamlit, que nao roda fora do servidor."""
    fake_st = mock.MagicMock()
    # Os decoradores de cache precisam devolver a funcao original, nao um MagicMock.
    fake_st.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    sys.modules["streamlit"] = fake_st
    # streamlit_agraph faz `import streamlit.components.v1`, e um MagicMock solto nao e
    # um pacote: os submodulos precisam estar registrados explicitamente.
    for submodulo in ("streamlit.components", "streamlit.components.v1"):
        sys.modules[submodulo] = mock.MagicMock()

    sys.path.insert(0, str(RAIZ))
    import utils  # noqa: E402

    utils.CURRENT_YEAR = ANO_BASE
    return utils


def _mock_upload(caminho: Path):
    """Reconstroi o objeto de arquivo que o Streamlit entrega aos processadores.

    O BOM e removido antes de entregar. Nao e maquiagem do oraculo: com o BOM, a primeira
    linha vira "\ufeffTY  - JOUR", o rispy nao a reconhece como tag e descarta o primeiro
    registro inteiro do arquivo — o app perde 2 documentos da base de exemplo sem avisar.
    A versao TypeScript corrige isso, e alinhar a entrada aqui faz a comparacao medir os
    algoritmos em vez de reproduzir o bug dos dois lados.
    """
    conteudo = caminho.read_bytes()
    if conteudo.startswith(b"\xef\xbb\xbf"):
        conteudo = conteudo[3:]
    buffer = io.BytesIO(conteudo)
    buffer.name = caminho.name
    return buffer


def _limpar(valor):
    """Converte tipos numpy/pandas em JSON e normaliza NaN para None."""
    if valor is None:
        return None
    if isinstance(valor, (bool, str)):
        return valor
    if isinstance(valor, (int,)):
        return int(valor)
    if isinstance(valor, float):
        return None if math.isnan(valor) or math.isinf(valor) else float(valor)
    # numpy escalares
    if hasattr(valor, "item"):
        try:
            return _limpar(valor.item())
        except (ValueError, AttributeError):
            pass
    return str(valor)


def _tabela(df, colunas: dict[str, str]) -> list[dict]:
    """Extrai TODAS as linhas de um DataFrame, renomeando as colunas.

    A tabela inteira, e nao um top-N: e o que permite comparar entidade por entidade em
    vez de por posicao, e detectar rotulos duplicados na tabela de venues.
    """
    if df is None or df.empty:
        return []

    linhas = []
    for _, linha in df.iterrows():
        linhas.append(
            {destino: _limpar(linha[origem]) for origem, destino in colunas.items() if origem in df}
        )
    return linhas


def main() -> None:
    utils = _carregar_utils()
    os.chdir(RAIZ)

    arquivos = [_mock_upload(RAIZ / nome) for nome in ARQUIVOS]
    df = utils.process_multiple_ris(arquivos, dict(ARQUIVOS))
    df = utils.padronizar_base_bibliometrica(df)
    print(f"base integrada: {len(df)} documentos", file=sys.stderr)

    df_doi, dup_doi = utils.deduplicar_por_doi(df)
    df_sim, dup_sim = utils.deduplicar_por_similaridade(df)

    metricas = utils.calcular_metricas_bibliometrix(df)
    resumo = utils.resumir_base_bibliometrica(df)
    completude = utils.analisar_completude_metadados(df)

    colunas_entidade = {
        "Qtd. de Citações": "citations",
        "Índice h": "h",
        "Índice g": "g",
        "Índice i10": "i10",
        "Índice m": "m",
        "Média de Citações": "meanCitations",
        "Mediana de Citações": "medianCitations",
        "Desvio Padrão de Citações": "stdCitations",
    }

    golden = {
        "_meta": {
            "baseYear": ANO_BASE,
            "arquivos": ARQUIVOS,
            "totalDocumentos": int(len(df)),
        },
        "dedupDoi": {
            "kept": int(len(df_doi)),
            "removed": int(len(dup_doi)),
        },
        "dedupSimilaridade": {
            "kept": int(len(df_sim)),
            "removed": int(len(dup_sim)),
        },
        "bibliometrix": {
            "growthRate": _limpar(metricas["growth_rate"]),
            "mcp": _limpar(metricas["mcp"]),
            "scp": _limpar(metricas["scp"]),
            "coauthIndex": _limpar(metricas["coauth_index"]),
            "singleAuthorDocs": _limpar(metricas["single_author_docs"]),
            "avgCitPerYear": _limpar(metricas["avg_cit_year"]),
        },
        "resumo": {
            "totalDocs": _limpar(resumo["total_docs"]),
            "timespan": _limpar(resumo["timespan"]),
            "avgAge": _limpar(resumo["avg_age"]),
            "authorsCount": _limpar(resumo["authors_count"]),
            "countriesCount": _limpar(resumo["countries_count"]),
            "keywordsCount": _limpar(resumo["kw_count"]),
            "venuesCount": _limpar(resumo["venues_count"]),
        },
        "completude": [
            {
                "field": linha["Metadado"],
                "missing": _limpar(linha["Faltantes"]),
                "missingPct": _limpar(linha["Faltantes (%)"]),
                "status": linha["Status"],
            }
            for _, linha in completude.iterrows()
        ],
        "tabelas": {
            "autores": _tabela(
                utils.gerar_tabela_autores(df), {"Autor": "entity", **colunas_entidade}
            ),
            "paises": _tabela(
                utils.gerar_tabela_paises(df), {"País": "entity", **colunas_entidade}
            ),
            "venues": _tabela(
                utils.gerar_tabela_venues(df),
                {"Local de Publicação (Venue)": "entity", **colunas_entidade},
            ),
            "keywords": _tabela(
                utils.gerar_tabela_keywords(df), {"Palavra-chave": "entity", **colunas_entidade}
            ),
        },
    }

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"golden.json escrito em {SAIDA}", file=sys.stderr)
    print(f"  dedup DOI:          {golden['dedupDoi']}", file=sys.stderr)
    print(f"  dedup similaridade: {golden['dedupSimilaridade']}", file=sys.stderr)
    print(
        "  tabelas: "
        + ", ".join(f"{k}={len(v)}" for k, v in golden["tabelas"].items()),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
