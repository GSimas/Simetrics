/**
 * Escalas e marcações de eixo dos gráficos SVG do Simetrics.
 *
 * Substituem o que o Plotly fazia por dentro: mapear domínio → pixels, achar marcas
 * "redondas" (1, 2, 5 × 10ⁿ) e formatar números no padrão brasileiro.
 */
import { numberLocale } from '@/lib/i18n/labels';
import type { Locale } from '@/lib/i18n/translations';

export interface Scale {
  (value: number): number;
  invert: (pixel: number) => number;
  domain: readonly [number, number];
  range: readonly [number, number];
}

export function linearScale(domain: readonly [number, number], range: readonly [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const scale = ((value: number) => r0 + ((value - d0) / span) * (r1 - r0)) as Scale;
  scale.invert = (pixel) => d0 + ((pixel - r0) / (r1 - r0 || 1)) * span;
  scale.domain = domain;
  scale.range = range;
  return scale;
}

/** Escala logarítmica (base 10). O domínio precisa ser positivo. */
export function logScale(domain: readonly [number, number], range: readonly [number, number]): Scale {
  const [d0, d1] = domain;
  const l0 = Math.log10(Math.max(d0, Number.MIN_VALUE));
  const l1 = Math.log10(Math.max(d1, Number.MIN_VALUE));
  const inner = linearScale([l0, l1], range);
  const scale = ((value: number) => inner(Math.log10(Math.max(value, Number.MIN_VALUE)))) as Scale;
  scale.invert = (pixel) => 10 ** inner.invert(pixel);
  scale.domain = domain;
  scale.range = range;
  return scale;
}

/** Passo "redondo" (1, 2, 5 × 10ⁿ) para cerca de `count` intervalos em `span`. */
export function niceStep(span: number, count: number): number {
  const raw = Math.abs(span) / Math.max(1, count);
  if (!Number.isFinite(raw) || raw === 0) return 1;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  return (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10) * power;
}

/** Domínio ampliado até múltiplos do passo, para as marcas caírem nas pontas. */
export function niceDomain(min: number, max: number, count = 5): [number, number] {
  if (min === max) {
    const pad = Math.abs(min) * 0.1 || 1;
    return niceDomain(min - pad, max + pad, count);
  }
  const step = niceStep(max - min, count);
  return [Math.floor(min / step) * step, Math.ceil(max / step) * step];
}

/** Marcas redondas dentro de [min, max]. */
export function linearTicks(min: number, max: number, count = 5): number[] {
  const step = niceStep(max - min, count);
  const start = Math.ceil(min / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let value = start; value <= max + step * 1e-9; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  return ticks;
}

/** Potências de 10 (e 2 e 5, quando há poucas décadas) dentro de [min, max]. */
export function logTicks(min: number, max: number): number[] {
  const low = Math.floor(Math.log10(Math.max(min, Number.MIN_VALUE)));
  const high = Math.ceil(Math.log10(Math.max(max, Number.MIN_VALUE)));
  const decades = high - low;
  const multipliers = decades <= 2 ? [1, 2, 5] : [1];
  const ticks: number[] = [];
  for (let exponent = low; exponent <= high; exponent += 1) {
    for (const multiplier of multipliers) {
      const value = multiplier * 10 ** exponent;
      if (value >= min * (1 - 1e-9) && value <= max * (1 + 1e-9)) ticks.push(Number(value.toPrecision(12)));
    }
  }
  return ticks;
}

/** Número no padrão do idioma (brasileiro por padrão); `digits` limita as casas decimais. */
export function formatNumber(value: number, digits = 2, locale: Locale = 'pt'): string {
  return value.toLocaleString(numberLocale(locale), { maximumFractionDigits: digits });
}

/** Rótulo curto de eixo: 1.200 → "1,2 mil", 3.400.000 → "3,4 mi" (em inglês, "k" e "M"). */
export function formatTick(value: number, locale: Locale = 'pt'): string {
  const abs = Math.abs(value);
  const en = locale === 'en';
  if (abs >= 1e6) return `${formatNumber(value / 1e6, 1, locale)}${en ? 'M' : ' mi'}`;
  if (abs >= 1e4) return `${formatNumber(value / 1e3, 1, locale)}${en ? 'k' : ' mil'}`;
  if (abs > 0 && abs < 0.01) return value.toExponential(0);
  return formatNumber(value, abs < 1 ? 2 : abs < 10 ? 1 : 0, locale);
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Semente fixa → sequência pseudoaleatória: o jitter dos pontos não muda a cada render. */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

/** Cor interpolada numa escala de várias paradas. */
export function interpolateColor(colors: readonly string[], t: number): string {
  if (colors.length === 0) return '#888';
  if (colors.length === 1) return colors[0] as string;
  const clamped = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  const position = clamped * (colors.length - 1);
  const index = Math.min(colors.length - 2, Math.floor(position));
  const local = position - index;
  const parse = (hex: string): number[] => [1, 3, 5].map((start) => parseInt(hex.slice(start, start + 2), 16));
  const a = parse(colors[index] as string);
  const b = parse(colors[index + 1] as string);
  const mix = a.map((value, channel) => Math.round(value + ((b[channel] as number) - value) * local));
  return `rgb(${mix.join(',')})`;
}
