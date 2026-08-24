import { useState } from 'react';
import { CheckCircle2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Textarea } from '@/components/ui/textarea';
import { useLocale } from '@/state/locale.store';

const FORM_NAME = 'simetrics-sus';

const SUS_STATEMENTS_PT = [
  'Acho que gostaria de usar o Simetrics com frequência.',
  'Considerei o Simetrics desnecessariamente complexo.',
  'Achei o Simetrics fácil de usar.',
  'Precisaria de apoio técnico para conseguir usar o Simetrics.',
  'Considerei que as funções do Simetrics estão bem integradas.',
  'Achei que havia inconsistência demais no Simetrics.',
  'Imagino que a maioria das pessoas aprenderia a usar o Simetrics rapidamente.',
  'Achei o Simetrics complicado de usar.',
  'Senti-me confiante ao usar o Simetrics.',
  'Precisei aprender muita coisa antes de conseguir usar o Simetrics.',
] as const;

const SUS_STATEMENTS_EN = [
  'I think that I would like to use Simetrics frequently.',
  'I found Simetrics unnecessarily complex.',
  'I thought Simetrics was easy to use.',
  'I think that I would need the support of a technical person to be able to use Simetrics.',
  'I found the various functions in Simetrics were well integrated.',
  'I thought there was too much inconsistency in Simetrics.',
  'I would imagine that most people would learn to use Simetrics very quickly.',
  'I found Simetrics very cumbersome to use.',
  'I felt very confident using Simetrics.',
  'I needed to learn a lot of things before I could get going with Simetrics.',
] as const;

const SCALE_LABELS_PT = ['Discordo totalmente', '', '', '', 'Concordo totalmente'] as const;
const SCALE_LABELS_EN = ['Strongly disagree', '', '', '', 'Strongly agree'] as const;

const TITULACAO_PT = [
  'Graduando(a)',
  'Graduado(a)',
  'Especialista',
  'Mestrando(a)',
  'Mestre',
  'Doutorando(a)',
  'Doutor(a)',
  'Pós-doutor(a)',
] as const;

const TITULACAO_EN = [
  'Undergraduate Student',
  'Bachelor Degree',
  'Specialist',
  "Master's Student",
  "Master's Degree",
  'Ph.D. Student',
  'Ph.D. / Doctorate',
  'Post-Doctorate',
] as const;

const AREAS_PT = [
  'Ciências Exatas e da Terra',
  'Ciências Biológicas',
  'Engenharias',
  'Ciências da Saúde',
  'Ciências Agrárias',
  'Ciências Sociais Aplicadas',
  'Ciências Humanas',
  'Linguística, Letras e Artes',
  'Multidisciplinar',
] as const;

const AREAS_EN = [
  'Exact and Earth Sciences',
  'Biological Sciences',
  'Engineering',
  'Health Sciences',
  'Agricultural Sciences',
  'Applied Social Sciences',
  'Humanities',
  'Linguistics, Literature & Arts',
  'Multidisciplinary',
] as const;

const EXPERIENCIA_PT = [
  'Nunca usei ferramentas bibliométricas',
  'Já usei uma ou duas vezes',
  'Uso ocasionalmente',
  'Uso com frequência',
] as const;

const EXPERIENCIA_EN = [
  'Never used bibliometric tools before',
  'Used once or twice',
  'Use occasionally',
  'Use frequently',
] as const;

const OPEN_QUESTIONS_PT = [
  { name: 'ux_navegacao', label: '11. Navegação e organização das abas' },
  { name: 'ux_visualizacao', label: '12. Clareza dos gráficos e visualizações' },
  { name: 'ux_ia', label: '13. Categorização temática e respostas da IA' },
  {
    name: 'ux_melhorias',
    label: '14. Se você fosse o engenheiro responsável, qual seria a primeira mudança?',
  },
  { name: 'ux_comentarios', label: '15. Comentários adicionais (opcional)' },
] as const;

const OPEN_QUESTIONS_EN = [
  { name: 'ux_navegacao', label: '11. Tab navigation and organization' },
  { name: 'ux_visualizacao', label: '12. Clarity of charts and visualizations' },
  { name: 'ux_ia', label: '13. Thematic categorization and AI responses' },
  {
    name: 'ux_melhorias',
    label: '14. If you were the lead engineer, what is the first change you would make?',
  },
  { name: 'ux_comentarios', label: '15. Additional comments (optional)' },
] as const;

export default function FeedbackTab() {
  const { locale, t } = useLocale();
  const isEn = locale === 'en';

  const [submitted, setSubmitted] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const susStatements = isEn ? SUS_STATEMENTS_EN : SUS_STATEMENTS_PT;
  const scaleLabels = isEn ? SCALE_LABELS_EN : SCALE_LABELS_PT;
  const titulacoes = isEn ? TITULACAO_EN : TITULACAO_PT;
  const areas = isEn ? AREAS_EN : AREAS_PT;
  const experiencias = isEn ? EXPERIENCIA_EN : EXPERIENCIA_PT;
  const openQuestions = isEn ? OPEN_QUESTIONS_EN : OPEN_QUESTIONS_PT;

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setSending(true);
    setError(null);

    const form = event.currentTarget;
    const data = new FormData(form);
    data.set('form-name', FORM_NAME);

    try {
      const response = await fetch('/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams(data as unknown as Record<string, string>).toString(),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setSubmitted(true);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? `${cause.message}. Local dev notice: Netlify Forms only registers on live production deploys.`
          : String(cause),
      );
    } finally {
      setSending(false);
    }
  };

  if (submitted) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle2 className="size-5 text-emerald-600" aria-hidden />
            {t('feedback_success_title')}
          </CardTitle>
          <CardDescription>
            {t('feedback_success_desc')}
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className="border-t-4 border-t-amber-500 shadow-xs">
      <CardHeader>
        <CardTitle className="text-base font-bold text-foreground">{t('feedback_title')}</CardTitle>
        <CardDescription>
          {t('feedback_desc')}
        </CardDescription>
      </CardHeader>

      <CardContent>
        <form
          name={FORM_NAME}
          method="POST"
          data-netlify="true"
          onSubmit={(event) => void handleSubmit(event)}
          className="space-y-8"
        >
          <input type="hidden" name="form-name" value={FORM_NAME} />

          <section className="space-y-4">
            <h3 className="text-sm font-semibold">{t('feedback_part1')}</h3>

            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label htmlFor="titulacao">{t('feedback_titulacao')}</Label>
                <Select name="titulacao" required>
                  <SelectTrigger id="titulacao">
                    <SelectValue placeholder={isEn ? 'Select...' : 'Selecione...'} />
                  </SelectTrigger>
                  <SelectContent>
                    {titulacoes.map((option) => (
                      <SelectItem key={option} value={option}>
                        {option}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="area">{t('feedback_area')}</Label>
                <Select name="area" required>
                  <SelectTrigger id="area">
                    <SelectValue placeholder={isEn ? 'Select...' : 'Selecione...'} />
                  </SelectTrigger>
                  <SelectContent>
                    {areas.map((option) => (
                      <SelectItem key={option} value={option}>
                        {option}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="experiencia">{t('feedback_experiencia')}</Label>
                <Select name="experiencia" required>
                  <SelectTrigger id="experiencia">
                    <SelectValue placeholder={isEn ? 'Select...' : 'Selecione...'} />
                  </SelectTrigger>
                  <SelectContent>
                    {experiencias.map((option) => (
                      <SelectItem key={option} value={option}>
                        {option}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </section>

          <Separator />

          <section className="space-y-4">
            <div>
              <h3 className="text-sm font-semibold">{t('feedback_part2')}</h3>
              <p className="text-sm text-muted-foreground">
                {t('feedback_part2_desc')}
              </p>
            </div>

            {susStatements.map((statement, index) => {
              const name = `sus_${String(index + 1).padStart(2, '0')}`;
              return (
                <fieldset key={name} className="space-y-2">
                  <legend className="text-sm">
                    {index + 1}. {statement}
                  </legend>
                  <RadioGroup name={name} required className="flex flex-wrap gap-4">
                    {[1, 2, 3, 4, 5].map((score) => (
                      <div key={score} className="flex items-center gap-1.5">
                        <RadioGroupItem value={String(score)} id={`${name}-${score}`} />
                        <Label htmlFor={`${name}-${score}`} className="text-xs font-normal">
                          {score}
                          {scaleLabels[score - 1] && (
                            <span className="ml-1 text-muted-foreground">
                              {scaleLabels[score - 1]}
                            </span>
                          )}
                        </Label>
                      </div>
                    ))}
                  </RadioGroup>
                </fieldset>
              );
            })}
          </section>

          <Separator />

          <section className="space-y-4">
            <h3 className="text-sm font-semibold">{t('feedback_part3')}</h3>

            {openQuestions.map((question) => (
              <div key={question.name} className="space-y-1.5">
                <Label htmlFor={question.name}>{question.label}</Label>
                <Textarea id={question.name} name={question.name} rows={3} />
              </div>
            ))}
          </section>

          {error && (
            <p className="rounded-md border border-destructive/40 bg-destructive/5 p-2 text-sm text-destructive">
              {error}
            </p>
          )}

          <Button
            type="submit"
            variant="gradient"
            disabled={sending}
            className="w-full font-semibold shadow-xs"
          >
            {sending ? (isEn ? 'Submitting...' : 'Enviando…') : t('feedback_submit_btn')}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
