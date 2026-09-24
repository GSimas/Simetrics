import type { CSSProperties } from 'react';
import { Pause } from 'lucide-react';

import { useStickyValue } from '@/lib/use-sticky-value';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { usePreferences } from '@/state/preferences.store';
import { useProjectStore } from '@/state/project.store';

/** Segundos por item: a velocidade de leitura fica igual com mais ou menos indicadores. */
const SECONDS_PER_ITEM = 5;

/**
 * Faixa rolante (news ticker) sob o cabeçalho, com o nome do projeto e os indicadores
 * da aba Informações Principais.
 *
 * O conteúdo aparece duas vezes em sequência e a faixa desliza metade da própria largura:
 * quando a primeira cópia sai pela esquerda, a segunda está exatamente onde ela começou,
 * e o laço não tem emenda. Clicar na faixa pausa ou retoma a rolagem — a mesma preferência
 * das Configurações, que fica salva. Sob o mouse ela também pausa, enquanto o cursor estiver
 * em cima.
 */
export function KpiTicker() {
  const { t, locale } = useLocale();
  const { value: overview } = useStickyValue(useDataset((state) => state.overview));
  const scroll = usePreferences((state) => state.tickerScroll);
  const projectName = useProjectStore(
    (state) => state.projects.find((project) => project.id === state.activeProjectId)?.name,
  );

  if (!overview) return null;

  const numberLocale = locale === 'en' ? 'en-US' : 'pt-BR';
  const format = (value: number): string =>
    value.toLocaleString(numberLocale, { maximumFractionDigits: 2 });
  const summary = overview.summary;
  const metrics = summary.bibliometrix;

  const items: [string, string][] = [
    [t('ticker_project'), projectName ?? t('ticker_unsaved')],
    [t('kpi_docs'), format(summary.totalDocs)],
    [t('kpi_docs_sub'), summary.timespan],
    [t('kpi_authors'), format(summary.authorsCount)],
    [t('kpi_countries'), format(summary.countriesCount)],
    [t('kpi_venues'), format(summary.venuesCount)],
    [t('kpi_growth'), `${format(metrics.growthRate)}%`],
    [t('kpi_citations_year'), format(metrics.avgCitPerYear)],
    [t('kpi_collab'), format(metrics.mcp)],
    [t('kpi_authors_doc'), format(metrics.coauthIndex)],
  ];

  const row = items.map(([label, value]) => (
    <span key={label} className="inline-flex shrink-0 items-center gap-2.5 px-5">
      <span className="eyebrow">{label}</span>
      <span className="text-sm font-medium tabular-nums text-foreground">{value}</span>
      <span className="ml-3 size-1 rounded-full bg-highlight" aria-hidden />
    </span>
  ));

  const toggle = (): void => usePreferences.getState().setTickerScroll(!scroll);

  return (
    <div
      className="ticker relative cursor-pointer overflow-hidden border-t border-border py-1.5 [mask-image:linear-gradient(90deg,transparent,#000_4%,#000_96%,transparent)]"
      role="button"
      aria-pressed={!scroll}
      aria-label={`${t('ticker_label')} — ${t('ticker_toggle')}`}
      title={t('ticker_toggle')}
      tabIndex={0}
      onClick={toggle}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          toggle();
        }
      }}
    >
      {/* Pausar congela a faixa onde está (animation-play-state), sem saltar ao início. */}
      <div
        className="ticker-track flex w-max"
        style={
          {
            '--ticker-duration': `${items.length * SECONDS_PER_ITEM}s`,
            animationPlayState: scroll ? undefined : 'paused',
          } as CSSProperties
        }
      >
        <div className="flex shrink-0">{row}</div>
        <div className="flex shrink-0" aria-hidden>
          {row}
        </div>
      </div>
      {!scroll && (
        <span className="eyebrow absolute right-3 top-1/2 flex -translate-y-1/2 items-center gap-1.5 bg-background px-2 py-0.5 text-foreground animate-in fade-in-0">
          <Pause className="size-3" aria-hidden />
          {t('ticker_paused')}
        </span>
      )}
    </div>
  );
}
