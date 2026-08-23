import Plotly from 'plotly.js/lib/core';
import bar from 'plotly.js/lib/bar';
import box from 'plotly.js/lib/box';
import choropleth from 'plotly.js/lib/choropleth';
import sankey from 'plotly.js/lib/sankey';
import scatter from 'plotly.js/lib/scatter';
import scatter3d from 'plotly.js/lib/scatter3d';
import scattergeo from 'plotly.js/lib/scattergeo';

/**
 * Bundle do Plotly montado sob medida.
 *
 * O pacote `plotly.js-dist-min` traz TODOS os tipos de traço — candlestick, carpet,
 * parcoords, mapas de contorno — e pesa 4,6 MB (1,38 MB comprimido) para desenhar
 * gráficos de barra. Registrando apenas o necessário, o mesmo chunk cai para 1,1 MB
 * (372 kB comprimido), e ainda é carregado sob demanda.
 *
 * **Ao acrescentar uma visualização, registre o traço aqui.** Sem o registro, o Plotly
 * ignora o traço em silêncio e desenha um gráfico vazio — sem erro no console e sem
 * exceção, o que torna a falha bem difícil de diagnosticar.
 *
 * | Traço        | Onde é usado                                             |
 * | :----------- | :------------------------------------------------------- |
 * | `bar`        | Produção por ano                                          |
 * | `scatter`    | Lei de Lotka, mapa temático, historiograph, mapa conceitual|
 * | `box`        | Distribuição estatística comparativa                      |
 * | `sankey`     | Fluxo de evolução temática                                |
 * | `choropleth` | Mapa de colaboração internacional                         |
 * | `scattergeo` | Arestas de colaboração sobre o mapa                       |
 * | `scatter3d`  | Mapa conceitual em três dimensões                         |
 *
 * O conjunto completo custa cerca de 630 kB comprimidos, contra 372 kB só com os três
 * primeiros. O chunk continua carregado sob demanda, então quem não abre uma aba com
 * gráfico não paga nada disso.
 */
Plotly.register([bar, box, choropleth, sankey, scatter, scatter3d, scattergeo]);

export default Plotly;
export type { Config, Data, Layout } from 'plotly.js';

/**
 * Tipo do traço da família scatter/bar/box.
 *
 * `Data` é uma união que inclui `indicator` e `gauge`, cujos `mode` aceitam apenas
 * `'gauge'`, `'delta'` e afins. Anotar um objeto literal como `Data` faz o TypeScript
 * resolver para o membro errado da união e rejeitar `'text+markers'`. `Partial<PlotData>`
 * nomeia o membro certo e continua atribuível a `Data`.
 */
export type Trace = Partial<import('plotly.js').PlotData>;
