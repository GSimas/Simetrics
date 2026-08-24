# Migração do Simetrics para React + Vite + TypeScript

Reimplementa a plataforma como aplicação Jamstack com deploy no Netlify. O app Streamlit
permanece na raiz do repositório, intocado, servindo como oráculo de paridade — nada foi
removido nesta migração.

## A restrição que moldou a arquitetura

As Netlify Functions têm limite de **10 s de execução e 6 MB de payload**. Isso torna
inviável hospedar o processamento no servidor, e um caso em particular era incompatível:
o assistente científico injetava a base inteira no prompt do Gemini. Medido nos dados de
exemplo, são **2,2 MB para 973 documentos**, e mais de **22 MB** no teto de 10.000.

A solução foi inverter a divisão de trabalho:

- **Navegador (Web Workers)** — ingestão, deduplicação, índices cientométricos, redes SNA,
  agrupamento temático e seleção de contexto por BM25.
- **Netlify Functions** — apenas a conversa com o Gemini, protegendo a `GEMINI_API_KEY`.

A cada pergunta, o worker seleciona os ~40 documentos relevantes e envia **48 kB** no
lugar dos 2,2 MB. A resposta volta em streaming, o que também resolve o timeout: o
primeiro byte sai em menos de um segundo.

## Paridade com o Python

Três scripts (`scripts/export_*.py`) executam as funções reais de `utils.py` e exportam
fixtures. **112 testes** comparam os dois lados.

| Verificação | Resultado |
| :--- | :--- |
| Ingestão RIS — 973 registros, todos os campos | idêntico |
| Deduplicação por DOI | idêntico |
| Índices h, g, i10, m — 70 distribuições | idêntico |
| Tabelas de autores (1.629), países (62), keywords (3.821) | idêntico |
| Métricas Bibliometrix, resumo, completude | idêntico |
| Grafo heterogêneo — 3.243 nós, 3.810 arestas | idêntico |
| Grau, closeness, betweenness exato, densidade, clustering, entropia | idêntico |
| Assortatividade, lei de potência, PageRank, autovetor | idêntico |
| `round()` do Python — 1.924 casos | idêntico |
| TF-IDF vs scikit-learn — vocabulário, idf, pesos, pares | idêntico |

## Nove defeitos corrigidos

Todos documentados em `web/README.md`, cada um com o teste que impede a regressão.

1. **Deduplicação por similaridade nunca executou.** `TfidfVectorizer(token_pattern=None)`
   sem `tokenizer` levanta `TypeError`, engolido por um `except Exception: pass`
   (`utils.py:2954`). O app só faz dedup por título exato, apesar do limiar de 0,90 na
   interface.
2. **A tabela de venues fragmenta revistas.** O agrupamento usa o valor bruto e só depois
   converte o rótulo para maiúsculas — 130 linhas duplicadas nos dados de exemplo.
3. **A contagem de venues do resumo repete a fragmentação**, anunciando 719 venues acima
   de uma tabela com 589 linhas.
4. **Referências citadas do Excel WoS se perdiam** — chave duplicada no dicionário de
   mapeamento (`utils.py:2279`).
5. **A centralidade de autovetor é sempre zero.** O NetworkX se recusa a calculá-la em
   grafo desconexo, e a exceção é engolida (`utils.py:2038`). Todo grafo bibliométrico
   real é desconexo — 49 componentes aqui.
6. **O betweenness muda a cada execução.** Amostragem sem semente: 33% dos nós mudam de
   valor entre duas rodadas seguidas, e o top-5 troca de ordem.
7. **O primeiro documento de todo arquivo com BOM é descartado** em silêncio — 2
   documentos perdidos nos dados de exemplo.
8. **A categorização temática levanta exceção** e nunca rodou, pelo mesmo
   `token_pattern=None`, desta vez sem `try/except`.
9. **O mapa conceitual agrupa por frequência, não por conceito.** Sem normalizar os
   vetores, 45 dos 50 termos caem num único grupo.

## Decisões técnicas que merecem atenção na revisão

**Arredondamento.** O `round()` do Python desempata para o dígito par sobre o valor binário
exato; o `toFixed()` do JavaScript desempata para longe do zero. Em 1.924 casos gerados do
Python, o `toFixed` erraria em 119 — e não em casos exóticos: 5 citações ÷ 8 documentos
= 0,625, que o Python arredonda para 0,62 e o JS para 0,63. Implementado em BigInt sobre a
mantissa (`src/core/stats.ts`).

**Autovetor precisa de deslocamento espectral.** O grafo heterogêneo é bipartido —
documentos só se ligam a autores, países e venues. O espectro fica simétrico (λ₂ = −λ₁), a
razão de convergência é exatamente 1, e a iteração de potência pura oscila
indefinidamente. Com 200 iterações o erro é 0,26; com 20.000 ainda não fecha. Iterar em
(A + σI) com σ do quociente de Rayleigh resolve em dezenas de iterações.

**Deduplicação por similaridade usa índice invertido.** Comparar todos os pares de 10.000
títulos seriam 50 milhões de operações. O índice invertido só visita pares que
compartilham um termo — 240 ms no teto de documentos.

**Betweenness escolhe a estratégia pelo tamanho do grafo.** Exato quando V×E cabe no
orçamento (136 ms nos dados de exemplo, contra 9,2 s do NetworkX), amostrado com semente
fixa acima disso. Ao contrário do original, o resultado é reproduzível.

## Desempenho

Pipeline completo em 10.000 documentos (~21 MB de RIS), via `npm run benchmark`:

| Etapa | Tempo |
| :--- | ---: |
| Ingestão (parse + normalização) | 255 ms |
| Deduplicação por similaridade | 240 ms |
| Quatro tabelas analíticas | 480 ms |
| Análise SNA completa (12.400 nós) | 1.580 ms |
| Mapa conceitual / mapa temático | ~110 ms cada |

## Como revisar

```bash
cd web && npm install
npm run test          # 112 testes, incluindo a paridade com o Python
npm run build
npm run dev           # interface
```

Para exercitar o assistente e a nomeação de temas localmente, em outro terminal:

```bash
GEMINI_API_KEY=sua-chave npx netlify-cli functions:serve --port 9999
```

Para regenerar os fixtures a partir do Python:

```bash
.venv/bin/python scripts/export_golden.py
.venv/bin/python scripts/export_primitives.py
.venv/bin/python scripts/export_graph_golden.py
```

## O que fica pendente

- **A geração de texto do Gemini não foi testada com chave real.** Todo o resto das
  functions está verificado — roteamento, validação, limite de payload, ausência de chave
  (503), chave inválida (502 genérico ao cliente, detalhe só no log).
- **O grafo circular de colaboração não foi confirmado visualmente.** Os dados e o layout
  têm cobertura de teste, e o mapa geográfico — que consome a mesma estrutura — está
  verificado no navegador.
- **Os `.ris` de demonstração estão duplicados** em `web/public/demo/`, somando 2,1 MB. É
  necessário enquanto o app Python permanece na raiz; some quando ele for removido.
- **Os clusters do K-Means não são reproduzíveis a partir do Python** — o `random_state`
  do scikit-learn depende do gerador do NumPy. São determinísticos entre execuções nossas.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
