import { useState } from 'react';
import { Check, Eye, EyeOff, Loader2, ShieldCheck, Trash2, X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { chatJson } from '@/lib/deepseek-client';
import { jevEvaluate } from '@/lib/jev-client';
import { normalizeHybridConfig, useHybridConfig, type HybridConfig } from '@/state/hybrid-config.store';
import { useLocale } from '@/state/locale.store';
import { HYBRID_COPY } from './copy';

interface HybridSettingsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

type TestState = { status: 'idle' } | { status: 'testing' } | { status: 'ok' } | { status: 'error'; message: string };

function SecretInput({ id, value, onChange, placeholder, label }: {
  id: string;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  label: string;
}) {
  const [visible, setVisible] = useState(false);
  return (
    <div className="relative">
      <Input
        id={id}
        type={visible ? 'text' : 'password'}
        value={value}
        autoComplete="off"
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        className="h-9 pr-10 font-mono text-xs"
      />
      <button
        type="button"
        onClick={() => setVisible((prev) => !prev)}
        aria-label={label}
        aria-pressed={visible}
        className="absolute right-3 top-2.5 text-muted-foreground transition-colors hover:text-foreground"
      >
        {visible ? <EyeOff className="size-4" aria-hidden /> : <Eye className="size-4" aria-hidden />}
      </button>
    </div>
  );
}

function TestResult({ state, okLabel }: { state: TestState; okLabel: string }) {
  if (state.status === 'ok') {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-emerald-700 dark:text-emerald-300">
        <Check className="size-3.5" aria-hidden /> {okLabel}
      </span>
    );
  }
  if (state.status === 'error') {
    return (
      <span className="inline-flex items-start gap-1 text-xs text-red-700 dark:text-red-300">
        <X className="mt-0.5 size-3.5 shrink-0" aria-hidden /> {state.message}
      </span>
    );
  }
  return null;
}

function NumberField({ id, label, value, onChange, step = 1, min, max }: {
  id: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  min: number;
  max: number;
}) {
  return (
    <div className="space-y-1">
      <Label htmlFor={id} className="text-[11px] text-muted-foreground">{label}</Label>
      <Input
        id={id}
        type="number"
        inputMode="decimal"
        value={Number.isFinite(value) ? value : ''}
        step={step}
        min={min}
        max={max}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-8 text-xs tabular-nums"
      />
    </div>
  );
}

export function HybridSettingsModal({ open, onOpenChange }: HybridSettingsModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto p-0">
        {/* O corpo só existe com o diálogo aberto: reabrir recomeça do que está salvo. */}
        <SettingsBody onClose={() => onOpenChange(false)} />
      </DialogContent>
    </Dialog>
  );
}

function SettingsBody({ onClose }: { onClose: () => void }) {
  const locale = useLocale((state) => state.locale);
  const copy = HYBRID_COPY[locale];
  const saved = useHybridConfig((state) => state.config);
  const setConfig = useHybridConfig((state) => state.setConfig);
  const clearKeys = useHybridConfig((state) => state.clearKeys);

  const [form, setForm] = useState<HybridConfig>(saved);
  const [generativeTest, setGenerativeTest] = useState<TestState>({ status: 'idle' });
  const [jevTest, setJevTest] = useState<TestState>({ status: 'idle' });

  const update = (patch: Partial<HybridConfig>) => setForm((prev) => ({ ...prev, ...patch }));

  const testGenerative = async () => {
    setGenerativeTest({ status: 'testing' });
    try {
      await chatJson(form.generative, {
        system: 'You answer with JSON only.',
        user: 'Reply with the JSON object {"ok": true}.',
      });
      setGenerativeTest({ status: 'ok' });
    } catch (cause) {
      setGenerativeTest({ status: 'error', message: cause instanceof Error ? cause.message : String(cause) });
    }
  };

  const testJev = async () => {
    setJevTest({ status: 'testing' });
    try {
      await jevEvaluate(
        {
          state: 'A randomized trial of vaccine efficacy in older adults.',
          model: form.jev.model,
          questions: { ok: { type: 'noul', instructions: 'Is this text about a scientific study?' } },
        },
        form.jev.apiKey,
      );
      setJevTest({ status: 'ok' });
    } catch (cause) {
      setJevTest({ status: 'error', message: cause instanceof Error ? cause.message : String(cause) });
    }
  };

  const save = () => {
    setConfig(normalizeHybridConfig(form));
    onClose();
  };

  return (
    <>
      <div className="space-y-5 p-6">
        <DialogHeader className="space-y-1.5">
          <DialogTitle className="text-xl font-bold">{copy.settingsTitle}</DialogTitle>
          <DialogDescription className="text-xs sm:text-sm">{copy.settingsSubtitle}</DialogDescription>
        </DialogHeader>

        <section className="space-y-3 rounded-xl border border-border/80 p-4" aria-labelledby="hybrid-gen-title">
          <h3 id="hybrid-gen-title" className="text-sm font-semibold">{copy.generativeSection}</h3>
          <div className="space-y-1">
            <Label htmlFor="hybrid-gen-key" className="text-xs">{copy.apiKeyOptional}</Label>
            <SecretInput
              id="hybrid-gen-key"
              value={form.generative.apiKey}
              onChange={(apiKey) => update({ generative: { ...form.generative, apiKey } })}
              placeholder="sk-..."
              label={copy.showKey}
            />
            <p className="text-[11px] text-muted-foreground">{copy.generativeKeyHint}</p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1">
              <Label htmlFor="hybrid-gen-model" className="text-xs">{copy.model}</Label>
              <Input
                id="hybrid-gen-model"
                value={form.generative.model}
                onChange={(event) => update({ generative: { ...form.generative, model: event.target.value } })}
                className="h-9 font-mono text-xs"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="hybrid-gen-url" className="text-xs">{copy.baseUrl}</Label>
              <Input
                id="hybrid-gen-url"
                value={form.generative.baseUrl}
                onChange={(event) => update({ generative: { ...form.generative, baseUrl: event.target.value } })}
                className="h-9 font-mono text-xs"
              />
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={generativeTest.status === 'testing' || !form.generative.apiKey.trim()}
              onClick={() => void testGenerative()}
            >
              {generativeTest.status === 'testing' ? <Loader2 className="animate-spin" aria-hidden /> : null}
              {generativeTest.status === 'testing' ? copy.testing : copy.test}
            </Button>
            <TestResult state={generativeTest} okLabel={copy.testOk} />
          </div>
        </section>

        <section className="space-y-3 rounded-xl border border-border/80 p-4" aria-labelledby="hybrid-jev-title">
          <h3 id="hybrid-jev-title" className="text-sm font-semibold">{copy.jevSection}</h3>
          <div className="grid gap-3 sm:grid-cols-[2fr_1fr]">
            <div className="space-y-1">
              <Label htmlFor="hybrid-jev-key" className="text-xs">{copy.apiKeyOptional}</Label>
              <SecretInput
                id="hybrid-jev-key"
                value={form.jev.apiKey}
                onChange={(apiKey) => update({ jev: { ...form.jev, apiKey } })}
                placeholder="ts-..."
                label={copy.showKey}
              />
              <p className="text-[11px] text-muted-foreground">{copy.jevKeyHint}</p>
            </div>
            <div className="space-y-1">
              <Label htmlFor="hybrid-jev-model" className="text-xs">{copy.model}</Label>
              <Input
                id="hybrid-jev-model"
                value={form.jev.model}
                onChange={(event) => update({ jev: { ...form.jev, model: event.target.value } })}
                className="h-9 font-mono text-xs"
              />
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={jevTest.status === 'testing'}
              onClick={() => void testJev()}
            >
              {jevTest.status === 'testing' ? <Loader2 className="animate-spin" aria-hidden /> : null}
              {jevTest.status === 'testing' ? copy.testing : copy.test}
            </Button>
            <TestResult state={jevTest} okLabel={copy.testOk} />
          </div>
        </section>

        <section className="space-y-3 rounded-xl border border-border/80 p-4" aria-labelledby="hybrid-params-title">
          <h3 id="hybrid-params-title" className="text-sm font-semibold">{copy.thresholdsSection}</h3>
          <div className="grid gap-3 sm:grid-cols-2">
            <NumberField
              id="hybrid-accept"
              label={copy.accept}
              value={form.thresholds.accept}
              step={0.05}
              min={0.05}
              max={1}
              onChange={(accept) => update({ thresholds: { ...form.thresholds, accept } })}
            />
            <NumberField
              id="hybrid-review"
              label={copy.reviewBelow}
              value={form.thresholds.review}
              step={0.05}
              min={0}
              max={1}
              onChange={(review) => update({ thresholds: { ...form.thresholds, review } })}
            />
          </div>

          <h3 className="pt-2 text-sm font-semibold">{copy.advanced}</h3>
          <div className="grid gap-3 sm:grid-cols-3">
            <NumberField id="hybrid-sample" label={copy.sampleSize} value={form.sampleSize} min={30} max={300} onChange={(sampleSize) => update({ sampleSize })} />
            <NumberField id="hybrid-max-cat" label={copy.maxCategories} value={form.maxCategories} min={3} max={30} onChange={(maxCategories) => update({ maxCategories })} />
            <NumberField id="hybrid-concurrency" label={copy.concurrency} value={form.concurrency} min={1} max={24} onChange={(concurrency) => update({ concurrency })} />
            <NumberField id="hybrid-clarify" label={copy.clarifyRounds} value={form.maxClarifyRounds} min={0} max={2} onChange={(maxClarifyRounds) => update({ maxClarifyRounds })} />
            <NumberField id="hybrid-expand" label={copy.expansionRounds} value={form.maxExpansionRounds} min={0} max={2} onChange={(maxExpansionRounds) => update({ maxExpansionRounds })} />
            <NumberField
              id="hybrid-leftover"
              label={copy.leftoverTrigger}
              value={Math.round(form.leftoverTrigger * 100)}
              min={2}
              max={50}
              onChange={(percent) => update({ leftoverTrigger: percent / 100 })}
            />
          </div>
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={form.reviewBeforeClassify}
              onChange={(event) => update({ reviewBeforeClassify: event.target.checked })}
              className="size-4 accent-[var(--highlight)]"
            />
            {copy.reviewBefore}
          </label>
        </section>

        <div className="flex items-start gap-2.5 rounded-xl border border-border/80 bg-muted/40 p-3 text-[11px] leading-relaxed text-muted-foreground">
          <ShieldCheck className="mt-0.5 size-4 shrink-0 text-emerald-600" aria-hidden />
          <span>{copy.privacy}</span>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2.5 border-t border-border/80 bg-muted/30 px-6 py-4">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => {
            clearKeys();
            setForm((prev) => ({ ...prev, generative: { ...prev.generative, apiKey: '' }, jev: { ...prev.jev, apiKey: '' } }));
          }}
          className="text-xs text-muted-foreground"
        >
          <Trash2 aria-hidden />
          {copy.clearKeys}
        </Button>
        <Button type="button" size="sm" onClick={save}>
          <Check aria-hidden />
          {copy.save}
        </Button>
      </div>
    </>
  );
}
