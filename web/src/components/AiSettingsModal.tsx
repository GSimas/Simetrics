import { useState } from 'react';
import {
  Check,
  ExternalLink,
  Eye,
  EyeOff,
  KeyRound,
  Loader2,
  ShieldCheck,
  Sparkles,
  Trash2,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { testAiConnection } from '@/lib/ai-client';
import {
  DEFAULT_MODELS,
  PROVIDER_MODELS,
  PROVIDER_OPTIONS,
  useAiConfig,
  type AiConfig,
  type AiProvider,
} from '@/state/ai-config.store';
import { useLocale } from '@/state/locale.store';

interface AiSettingsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AiSettingsModal({ open, onOpenChange }: AiSettingsModalProps) {
  const { t, locale } = useLocale();
  const isEn = locale === 'en';

  const globalConfig = useAiConfig((state) => state.config);
  const setGlobalConfig = useAiConfig((state) => state.setConfig);
  const clearGlobalConfig = useAiConfig((state) => state.clearConfig);

  const [form, setForm] = useState<AiConfig>(globalConfig);
  const [showKey, setShowKey] = useState(false);
  const [testing, setTesting] = useState(false);
  const [isCustomModel, setIsCustomModel] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);

  const selectedProviderOption =
    PROVIDER_OPTIONS.find((opt) => opt.id === form.provider) ?? PROVIDER_OPTIONS[0]!;

  const availableModels = PROVIDER_MODELS[form.provider] ?? [];
  const isKnownModel = availableModels.some((m) => m.id === form.model);

  const handleProviderChange = (provider: AiProvider) => {
    const defaultModel = DEFAULT_MODELS[provider];
    setForm((prev) => ({
      ...prev,
      provider,
      model: defaultModel,
    }));
    setIsCustomModel(false);
    setTestResult(null);
  };

  const handleModelSelectChange = (value: string) => {
    if (value === '__custom__') {
      setIsCustomModel(true);
    } else {
      setIsCustomModel(false);
      setForm((prev) => ({ ...prev, model: value }));
    }
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      await testAiConnection(form);
      setTestResult({
        success: true,
        message: isEn ? 'Connection tested successfully!' : 'Conexão realizada com sucesso!',
      });
    } catch (err) {
      setTestResult({
        success: false,
        message: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = () => {
    setGlobalConfig(form);
    onOpenChange(false);
  };

  const handleClear = () => {
    clearGlobalConfig();
    setForm({ provider: 'gemini', apiKey: '', model: DEFAULT_MODELS.gemini });
    setIsCustomModel(false);
    setTestResult(null);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl overflow-hidden p-0 border-border/80 bg-card shadow-2xl rounded-2xl">
        <div className="h-1.5 w-full bg-gradient-to-r from-purple-600 via-indigo-500 to-blue-600" />

        <div className="p-6 space-y-5">
          <DialogHeader className="space-y-1.5">
            <div className="flex items-center gap-2 text-purple-600 dark:text-purple-400 font-semibold text-xs uppercase tracking-wider">
              <KeyRound className="size-4" />
              <span>Bring Your Own Key (BYOK)</span>
            </div>
            <DialogTitle className="text-xl font-bold text-foreground">
              {t('ai_modal_title')}
            </DialogTitle>
            <DialogDescription className="text-xs sm:text-sm text-muted-foreground">
              {t('ai_modal_subtitle')}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            {/* 1. Provedor de IA */}
            <div className="space-y-1.5">
              <Label htmlFor="ai-provider" className="text-xs font-semibold text-foreground">
                {t('ai_provider_label')}
              </Label>
              <Select value={form.provider} onValueChange={(val) => handleProviderChange(val as AiProvider)}>
                <SelectTrigger id="ai-provider" className="h-10 rounded-xl">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PROVIDER_OPTIONS.map((opt) => (
                    <SelectItem key={opt.id} value={opt.id}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* 2. Chave de API */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label htmlFor="ai-key" className="text-xs font-semibold text-foreground">
                  {t('ai_api_key_label')}
                </Label>
                <a
                  href={selectedProviderOption.helpUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-[11px] font-medium text-primary hover:underline"
                >
                  <span>{isEn ? 'Get API Key' : 'Obter chave de API'}</span>
                  <ExternalLink className="size-3" />
                </a>
              </div>
              <div className="relative">
                <Input
                  id="ai-key"
                  type={showKey ? 'text' : 'password'}
                  value={form.apiKey}
                  onChange={(e) => setForm((prev) => ({ ...prev, apiKey: e.target.value }))}
                  placeholder={selectedProviderOption.placeholder}
                  className="pr-10 h-10 font-mono text-xs rounded-xl"
                />
                <button
                  type="button"
                  onClick={() => setShowKey((prev) => !prev)}
                  className="absolute right-3 top-3 text-muted-foreground hover:text-foreground transition-colors"
                  tabIndex={-1}
                >
                  {showKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                </button>
              </div>
            </div>

            {/* 3. Seleção do Modelo LLM (Lista Suspensa por Provedor) */}
            <div className="space-y-2">
              <div className="space-y-1.5">
                <Label htmlFor="ai-model-select" className="text-xs font-semibold text-foreground">
                  {t('ai_model_label')}
                </Label>
                <Select
                  value={isCustomModel || !isKnownModel ? '__custom__' : form.model}
                  onValueChange={handleModelSelectChange}
                >
                  <SelectTrigger id="ai-model-select" className="h-10 rounded-xl">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {availableModels.map((opt) => (
                      <SelectItem key={opt.id} value={opt.id}>
                        <div className="flex items-center justify-between gap-2 w-full">
                          <span>{opt.name}</span>
                          {opt.badge && (
                            <span className="rounded bg-primary/10 px-1.5 py-0.2 text-[10px] font-semibold text-primary">
                              {opt.badge}
                            </span>
                          )}
                        </div>
                      </SelectItem>
                    ))}
                    <SelectItem value="__custom__">
                      <span className="italic text-muted-foreground">
                        {isEn ? '✏️ Other custom model...' : '✏️ Outro modelo personalizado...'}
                      </span>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Campo para Digitar Modelo Personalizado (se selecionado "Outro") */}
              {(isCustomModel || !isKnownModel) && (
                <div className="pt-1 space-y-1 animate-in fade-in-0">
                  <Label htmlFor="ai-custom-model" className="text-[11px] text-muted-foreground">
                    {isEn ? 'Enter custom model identifier:' : 'Digite o identificador do modelo:'}
                  </Label>
                  <Input
                    id="ai-custom-model"
                    value={form.model}
                    onChange={(e) => setForm((prev) => ({ ...prev, model: e.target.value }))}
                    placeholder="ex: gemini-1.5-pro-exp-0827 / gpt-4.5-preview / deepseek-r1"
                    className="h-9 font-mono text-xs rounded-xl"
                  />
                </div>
              )}
            </div>

            {/* 4. URL Base Customizada (para Ollama / LM Studio / Proxy Local) */}
            {form.provider === 'custom' && (
              <div className="space-y-1.5 animate-in fade-in-0">
                <Label htmlFor="ai-base-url" className="text-xs font-semibold text-foreground">
                  {t('ai_base_url_label')}
                </Label>
                <Input
                  id="ai-base-url"
                  value={form.baseUrl ?? ''}
                  onChange={(e) => setForm((prev) => ({ ...prev, baseUrl: e.target.value }))}
                  placeholder="http://localhost:11434/v1"
                  className="h-10 font-mono text-xs rounded-xl"
                />
                <p className="text-[11px] text-muted-foreground">
                  {isEn
                    ? 'Default for Ollama is http://localhost:11434/v1, LM Studio is http://localhost:1234/v1'
                    : 'Padrão para Ollama é http://localhost:11434/v1 e LM Studio é http://localhost:1234/v1'}
                </p>
              </div>
            )}

            {/* Resultado do Teste */}
            {testResult && (
              <div
                className={`rounded-xl p-3 text-xs flex items-start gap-2 border animate-in fade-in-0 ${
                  testResult.success
                    ? 'border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/60 dark:text-emerald-300'
                    : 'border-red-200 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/60 dark:text-red-300'
                }`}
              >
                {testResult.success ? (
                  <Check className="size-4 shrink-0 mt-0.5 text-emerald-600" />
                ) : (
                  <ShieldCheck className="size-4 shrink-0 mt-0.5 text-red-600" />
                )}
                <span>{testResult.message}</span>
              </div>
            )}

            {/* Nota de Privacidade */}
            <div className="rounded-xl border border-border/80 bg-muted/40 p-3 flex items-start gap-2.5 text-[11px] leading-relaxed text-muted-foreground">
              <ShieldCheck className="size-4 shrink-0 text-emerald-600 mt-0.5" />
              <span>{t('ai_privacy_note')}</span>
            </div>
          </div>
        </div>

        {/* Rodapé de Ações */}
        <div className="border-t border-border/80 bg-muted/30 px-6 py-4 flex flex-wrap items-center justify-between gap-2.5">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={handleClear}
            className="text-xs text-muted-foreground hover:bg-red-50 hover:text-red-700 dark:hover:bg-red-950 dark:hover:text-red-300"
          >
            <Trash2 className="size-3.5" />
            {t('ai_clear_btn')}
          </Button>

          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => void handleTest()}
              disabled={testing || (!form.apiKey && form.provider !== 'custom')}
              className="text-xs font-medium"
            >
              {testing ? (
                <>
                  <Loader2 className="size-3.5 animate-spin" />
                  {isEn ? 'Testing...' : 'Testando...'}
                </>
              ) : (
                <>
                  <Sparkles className="size-3.5 text-purple-600" />
                  {t('ai_test_btn')}
                </>
              )}
            </Button>

            <Button
              type="button"
              variant="gradient"
              size="sm"
              onClick={handleSave}
              className="text-xs font-semibold shadow-xs"
            >
              <Check className="size-3.5" />
              {t('ai_save_btn')}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function AiSettingsButton({ onClick }: { onClick: () => void }) {
  const t = useLocale((state) => state.t);
  const isConfigured = useAiConfig((state) => state.isConfigured());

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onClick}
      className={`h-9 gap-1.5 rounded-xl border text-xs font-semibold shadow-2xs transition-all ${
        isConfigured
          ? 'border-purple-300/80 bg-purple-50/70 text-purple-800 hover:bg-purple-100 dark:border-purple-900/60 dark:bg-purple-950/60 dark:text-purple-300'
          : 'border-border/80 bg-card/80 text-foreground hover:bg-muted'
      }`}
      title={isConfigured ? t('ai_configured') : t('ai_not_configured')}
    >
      <KeyRound
        className={`size-3.5 ${isConfigured ? 'text-purple-600 dark:text-purple-400' : 'text-muted-foreground'}`}
        aria-hidden
      />
      <span>{t('ai_settings_btn')}</span>
      {isConfigured && <span className="size-1.5 rounded-full bg-emerald-500 animate-pulse" />}
    </Button>
  );
}
