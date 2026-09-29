import { FileText, LayoutDashboard, Layers, Network } from 'lucide-react';

import { lazyWithPreload } from '@/lib/lazy';
import type { TranslationKey } from '@/lib/i18n/translations';
import type { BibliometricView } from '@/state/navigation.store';

/**
 * As vistas da aba Análise Bibliométrica. Cada uma continua sendo um chunk próprio: abrir
 * a aba baixa só a vista aberta; as outras vêm no ócio (`preloadBibliometricViews`).
 */
export const BIBLIOMETRIC_VIEWS = [
  {
    value: 'overview',
    labelKey: 'tab_overview',
    Icon: LayoutDashboard,
    Panel: lazyWithPreload(() => import('@/features/overview/OverviewTab')),
  },
  {
    value: 'networks',
    labelKey: 'tab_networks',
    Icon: Network,
    Panel: lazyWithPreload(() => import('@/features/networks/NetworksTab')),
  },
  {
    value: 'advanced',
    labelKey: 'tab_advanced',
    Icon: Layers,
    Panel: lazyWithPreload(() => import('@/features/overview/AdvancedTab')),
  },
  {
    value: 'report',
    labelKey: 'tab_report',
    Icon: FileText,
    Panel: lazyWithPreload(() => import('@/features/report/ReportTab')),
  },
] as const satisfies readonly { value: BibliometricView; labelKey: TranslationKey; Icon: unknown; Panel: unknown }[];

export function preloadBibliometricViews(): void {
  for (const view of BIBLIOMETRIC_VIEWS) void view.Panel.preload();
}
