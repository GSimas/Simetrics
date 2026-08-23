# Simetrics Web

Migração do Simetrics (Streamlit/Python) para React + Vite + TypeScript, com deploy no Netlify.

## Arquitetura

O processamento pesado — deduplicação, índices cientométricos, similaridade de Jaccard,
clustering e métricas de rede — roda **inteiramente no cliente, dentro de Web Workers**.
As Netlify Functions existem apenas para intermediar a API do Gemini e proteger a
`GEMINI_API_KEY`, respeitando o limite de 10s de execução e 6MB de payload da plataforma.

```
src/core/      lógica pura, sem React — é o que os testes de paridade exercitam
src/workers/   fronteira assíncrona (Comlink) entre a UI e o core
src/features/  uma pasta por aba da interface
netlify/       funções serverless (somente Gemini)
```

## Desenvolvimento

```bash
npm install
npm run dev
```

Comandos: `build`, `typecheck`, `lint`, `test`, `test:parity`, `benchmark`.

As funções serverless não sobem com o `vite dev`. Para exercitar o assistente e a
nomeação de temas localmente, rode em outro terminal:

```bash
npx netlify-cli functions:serve --port 9999
```

O `vite.config.ts` já encaminha `/api` para essa porta. Alternativamente, `netlify dev`
sobe os dois lados de uma vez. Sem nenhum dos dois, as chamadas de IA falham — e a
interface trata isso como qualquer indisponibilidade, sem quebrar.

## Variáveis de ambiente

`GEMINI_API_KEY` é obrigatória e fica no painel do Netlify. **Nunca** prefixe com `VITE_` —
variáveis com esse prefixo são embutidas no bundle do cliente e a chave vazaria.

`GEMINI_CHAT_MODEL` e `GEMINI_LABEL_MODEL` são opcionais e trocam os modelos sem novo
deploy do código, útil quando um identificador é descontinuado.

## Inteligência artificial

Duas funcionalidades, ambas com o processamento pesado no navegador e a rede reservada
apenas ao que exige o modelo.

**Mapeamento temático.** TF-IDF → LSA (SVD randomizado) → K-Means, com o número de temas
escolhido pelo Silhouette. Tudo isso roda no worker; só as amostras de cada agrupamento
vão ao modelo, que devolve o nome do tema — uma requisição por tema. O laço equivalente no
Python leva ~25 s numa chamada só, acima do limite de 10 s da Netlify. Quando a nomeação
falha, o tema recebe um rótulo derivado dos próprios termos característicos: perder o nome
não pode custar o agrupamento, que é a parte cara.

**Assistente científico.** A cada pergunta, o worker seleciona por BM25 os ~40 documentos
mais relevantes e envia só eles, junto com um panorama agregado da base. Medido na base de
exemplo: **48 kB contra 2,2 MB** da base inteira, e mais de 22 MB no teto de 10.000
documentos — bem acima do limite de 6 MB de payload da plataforma. A resposta volta em
streaming, o que também resolve o timeout: o primeiro byte sai em menos de um segundo.

O agregado acompanha o recorte porque perguntas globais — "quem mais publica nesta área?" —
não se respondem a partir de quarenta documentos.

## Paridade com o Streamlit

O app Python permanece na raiz do repositório como oráculo. Regenere os fixtures sempre
que o Python mudar:

```bash
.venv/bin/python scripts/export_golden.py      # pipeline de ponta a ponta
.venv/bin/python scripts/export_primitives.py  # arredondamento, title case, índices, TF-IDF
.venv/bin/python scripts/export_graph_golden.py # métricas de rede (NetworkX)
```

`npm run test:parity` compara os dois. Estado atual sobre os três .ris de exemplo
(973 documentos):

| Verificação | Resultado |
| :--- | :--- |
| Ingestão RIS (973 registros, todos os campos) | idêntico |
| Deduplicação por DOI | idêntico |
| Métricas Bibliometrix, resumo, completude | idêntico |
| Tabela de autores (1629 entidades) | idêntico |
| Tabela de países (62) | idêntico |
| Tabela de keywords (3821) | idêntico |
| Tabela de venues | reagrupada — ver abaixo |
| Contagem de venues no resumo | reagrupada — ver abaixo |
| Grafo heterogêneo (3.243 nós, 3.810 arestas) | idêntico |
| Grau, centralidade de grau, closeness | idêntico |
| Betweenness exato (Brandes) | idêntico |
| Densidade, clustering, entropia, eficiência global | idêntico |
| Assortatividade, lei de potência, PageRank | idêntico |
| Autovetor | idêntico no maior componente — ver abaixo |
| Clusters temáticos (K-Means) | não comparável (semente do sklearn) |

## Divergências deliberadas

Pontos onde o TypeScript não reproduz o Python, porque o Python está errado. Cada um
está documentado no código e coberto por teste.

1. **Deduplicação por similaridade nunca rodou.** `deduplicar_por_similaridade` monta o
   `TfidfVectorizer` com `token_pattern=None` e sem `tokenizer`, o que levanta `TypeError`
   no scikit-learn. A chamada está dentro de um `except Exception: pass` (utils.py:2954),
   então a falha é silenciosa e o app só faz dedup por título exato. Aqui a etapa funciona.

2. **A tabela de venues fragmenta revistas.** `gerar_tabela_venues` agrupa pelo valor bruto
   e só converte para maiúsculas no rótulo, então a mesma revista com capitalização
   diferente entre bases vira várias linhas com o mesmo nome — 130 duplicadas nos dados de
   exemplo. Aqui a normalização acontece antes do agrupamento.

3. **A contagem de venues do resumo repete a fragmentação.** O `nunique()` de
   `resumir_base_bibliometrica` opera sobre o valor bruto da coluna, pela mesma razão da
   tabela. Sem normalizar antes de contar, o painel anunciaria 719 venues logo acima de
   uma tabela com 589 linhas.

4. **Referências citadas do Excel WoS se perdiam.** O dicionário de mapeamento repete a
   chave `'Cited References'` (utils.py:2279), e em Python a segunda entrada sobrescreve a
   primeira, então a coluna `CITED REFERENCES` nunca era criada. Aqui os dois destinos são
   preenchidos.

5. **A centralidade de autovetor é sempre zero.** `nx.eigenvector_centrality_numpy`
   levanta `AmbiguousSolution` em grafo desconexo — e todo grafo bibliométrico real é
   desconexo (49 componentes nos dados de exemplo). O app captura a exceção e preenche a
   coluna inteira com zero (utils.py:2038). Aqui usamos iteração de potência com
   deslocamento espectral, que devolve o autovetor dominante. Validado contra o NetworkX
   rodando no maior componente isolado, onde ele calcula normalmente.

6. **O betweenness muda a cada execução.** O app chama
   `nx.betweenness_centrality(G, k=100)` sem `seed`, o que sorteia as fontes pelo estado
   global do `random`. Medido nos dados de exemplo: 33% dos nós mudam de valor entre duas
   rodadas seguidas, e o próprio top-5 troca de ordem — apesar do comentário no código
   prometer "99% de precisão". Aqui o Brandes é exato quando o grafo cabe no orçamento
   (V×E ≤ 20M) e, acima disso, amostrado com semente fixa. Nos dados de exemplo o
   resultado é exato e leva 136 ms, contra 9,2 s do NetworkX.

7. **O primeiro documento de todo arquivo com BOM é descartado.** `scielo.ris` e
   `wos.ris` começam com BOM (`EF BB BF`). O app decodifica com
   `io.StringIO(bytes.decode("utf-8"))`, que preserva o BOM, e a primeira linha vira
   `\uFEFFTY  - JOUR` — que não casa com o padrão de tag do RIS. O rispy segue procurando
   um `TY` e engole o registro inteiro até o próximo. São 2 documentos perdidos em
   silêncio só na base de exemplo, de 973 para 971.

   Este foi encontrado por uma divergência entre navegador e Node: o `TextDecoder` do
   navegador remove o BOM por padrão, o `Buffer.toString('utf8')` do Node e o Python o
   mantêm. Os scripts de exportação do oráculo removem o BOM antes de entregar ao rispy,
   para que a comparação meça os algoritmos em vez de reproduzir o bug dos dois lados.

8. **O mapa conceitual agrupa por frequência, não por conceito.** `gerar_mapas_conceituais`
   roda o K-Means sobre as contagens brutas da matriz termo × documento. Como o
   comprimento do vetor é proporcional à frequência do termo, o agrupamento separa por
   QUÃO COMUM o termo é em vez de COM QUEM ele aparece. Medido na base de exemplo: os
   quatro grupos saem com 3, 1, 45 e 1 termos — uma mancha só. Normalizando os vetores
   antes do agrupamento, saem 2, 23, 20 e 5, correspondendo a teoria memética, algoritmos
   de otimização, cultura e cognição, e comunicação digital.

Além disso, `categorizar_temas_por_cluster` (utils.py:1176) tem o mesmo defeito de
`token_pattern=None`, mas sem `try/except` — ou seja, a categorização temática por IA
levanta exceção no app atual. Será reimplementada na Fase 5.

## Grafos

Dois grafos, com propósitos distintos:

- **Heterogêneo** — documentos ligados a autores, países e venues. É a base das métricas
  de ecologia profunda. Note que ele é **bipartido**: documentos nunca se ligam entre si.
  Isso torna o espectro simétrico (λ₂ = −λ₁) e faz a iteração de potência pura não
  convergir, daí o deslocamento espectral em `eigenvectorCentrality`.
- **Coocorrência** — entidades do mesmo tipo ligadas por aparecerem no mesmo documento,
  com peso pela frequência. É a rede que o usuário visualiza, recortada por top-N e
  colorida pelas comunidades do Louvain (no lugar do `greedy_modularity_communities`).

Os algoritmos de centralidade rodam sobre uma representação CSR com arrays tipados, e não
sobre a API de objetos do Graphology — o laço interno de Brandes percorre a vizinhança
milhões de vezes, e o custo de hashing dominaria. O Graphology continua responsável pelo
modelo de grafo, pelo Louvain e pelo layout.

## Desempenho

`npm run benchmark` roda o pipeline no teto de 10.000 documentos (~21 MB de RIS).
Referência em um MacBook, tempos aproximados:

| Etapa | Tempo |
| :--- | ---: |
| Ingestão (parse + normalização) | 255 ms |
| Deduplicação por DOI | 16 ms |
| Deduplicação por similaridade | 240 ms |
| Resumo + completude | 70 ms |
| Quatro tabelas analíticas | 480 ms |
| Betweenness amostrado (12.400 nós) | 40 ms |
| Análise SNA completa (12.400 nós) | 1.580 ms |
| Rede de coautoria, top 50 | 20 ms |
| Mapa conceitual (PCA, 50 termos) | 100 ms |
| Mapa temático (150 termos) | 115 ms |
| Sankey, genética, colaboração, boxplot | menos de 25 ms cada |

A deduplicação por similaridade usa índice invertido em vez de produto matricial: comparar
todos os pares de 10.000 títulos seriam 50 milhões de operações.

O betweenness escolhe a estratégia pelo tamanho do grafo. Nos dados de exemplo (3.239 nós)
ele roda exato em 136 ms — o NetworkX leva 9,2 s para o mesmo cálculo. Em 10.000
documentos o grafo chega a 12.400 nós e 40.338 arestas, onde o exato custaria 4 s, e a
amostragem semeada entra no lugar.

## Interface

Cinco abas, em Radix Tabs, com os textos em pt-BR do app original. Todo o processamento é
disparado por interação e memorizado no store: trocar de aba nunca recalcula.

As visualizações mais pesadas ficam em sub-abas dentro de **Análises visuais** (Visão
Geral) e **Colaboração internacional** (Redes). Cada sub-aba só calcula quando é aberta —
empilhá-las dispararia sete varreduras completas da base de uma vez.

| Visualização | O que responde |
| :--- | :--- |
| Distribuição (boxplot) | Como a métrica se dispersa entre até 5 entidades, com outliers |
| Evolução temática (Sankey) | Quais termos sobrevivem de um período ao seguinte |
| Genética das ideias | Quando cada termo nasceu, por quanto tempo durou, quanto replicou |
| Mapa conceitual (PCA 2D/3D) | Quais termos habitam os mesmos documentos |
| Mapa temático | Centralidade × densidade, nos quatro quadrantes do Bibliometrix |
| Historiograph | Quem cita quem dentro da própria base |
| Colaboração internacional | Geografia das parcerias, no mapa e em grafo circular |

O código é dividido por rota de carregamento — Plotly, Sigma e a nuvem de palavras entram
em `React.lazy`, então quem abre o app sem carregar base baixa apenas o chunk principal.

## Notas de dependências

- **SheetJS (`xlsx`)** vem do CDN oficial, não do npm: a versão publicada no npm está
  parada em 0.18.5 e carrega advisories de prototype pollution e ReDoS sem correção.
  É a única lib que lê o formato `.xls` binário legado, exigido pelos exports do WoS.
- **Nuvem de palavras** usa `d3-cloud` para o layout em vez de `echarts-wordcloud`, cujo
  peer é preso ao ECharts 5 — e o ECharts 5 tem um advisory de XSS aberto. A renderização
  é em SVG, então o texto continua selecionável e a exportação sai vetorial.
- **Plotly** é montado sob medida em `src/components/charts/plotly.ts`, registrando apenas
  os traços usados. O `plotly.js-dist-min` pré-empacotado traz todos e pesa 4,6 MB
  (1,38 MB comprimido); o bundle atual tem 1,1 MB (372 kB). Ao adicionar uma visualização,
  registre o traço lá — sem registro o Plotly desenha um gráfico vazio, em silêncio.
- **`@tanstack/react-table` fica na v8**, e não na v9. A v9 reescreveu a API em torno de
  features opt-in (`useTable`, `tableFeatures`, `createCoreRowModel`), e a tabela aqui é
  componente de apoio: não vale trocar uma API estável e documentada por outra que
  exigiria engenharia reversa dos tipos.
- **`global: 'globalThis'`** no `vite.config.ts` é obrigatório. O `plotly.js` de
  código-fonte arrasta dependências que referenciam o `global` do Node; sem o mapeamento a
  aplicação quebra em execução, e apenas em execução — o build conclui sem reclamar.
