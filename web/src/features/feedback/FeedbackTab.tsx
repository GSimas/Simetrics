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

/**
 * Avaliação de usabilidade (SUS) — ⇄ o formulário de Geral.py:2292.
 *
 * As respostas vão para o Netlify Forms, no lugar do Google Sheets. O envio é um POST
 * `application/x-www-form-urlencoded` para a própria página; o Netlify intercepta pelo
 * campo `form-name`, então o formulário precisa existir no HTML estático do build para
 * ser detectado — daí o formulário oculto em `index.html`.
 */

const FORM_NAME = 'simetrics-sus';

/** As 10 afirmações do System Usability Scale, na ordem canônica. */
const SUS_STATEMENTS = [
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

const SCALE_LABELS = ['Discordo totalmente', '', '', '', 'Concordo totalmente'] as const;

const TITULACAO = [
  'Graduando(a)',
  'Graduado(a)',
  'Especialista',
  'Mestrando(a)',
  'Mestre',
  'Doutorando(a)',
  'Doutor(a)',
  'Pós-doutor(a)',
] as const;

const AREAS = [
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

const EXPERIENCIA = [
  'Nunca usei ferramentas bibliométricas',
  'Já usei uma ou duas vezes',
  'Uso ocasionalmente',
  'Uso com frequência',
] as const;

const OPEN_QUESTIONS = [
  { name: 'ux_navegacao', label: '11. Navegação e organização das abas' },
  { name: 'ux_visualizacao', label: '12. Clareza dos gráficos e visualizações' },
  { name: 'ux_ia', label: '13. Categorização temática e respostas da IA' },
  {
    name: 'ux_melhorias',
    label: '14. Se você fosse o engenheiro responsável, qual seria a primeira mudança?',
  },
  { name: 'ux_comentarios', label: '15. Comentários adicionais (opcional)' },
] as const;

export default function FeedbackTab() {
  const [submitted, setSubmitted] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

      if (!response.ok) throw new Error(`O envio falhou (HTTP ${response.status}).`);
      setSubmitted(true);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? `${cause.message} Em ambiente de desenvolvimento local isso é esperado: o Netlify Forms só responde no site publicado.`
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
            Avaliação registrada
          </CardTitle>
          <CardDescription>
            Obrigado por dedicar seu tempo. Suas respostas orientam diretamente as
            próximas melhorias da plataforma.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Avaliação da Plataforma</CardTitle>
        <CardDescription>
          Este questionário avalia a sua experiência com o Simetrics. Não há respostas
          certas ou erradas — estamos avaliando o sistema, não você. As respostas são
          anônimas.
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
            <h3 className="text-sm font-semibold">Parte 1 · Perfil do participante</h3>

            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label htmlFor="titulacao">Titulação</Label>
                <Select name="titulacao" required>
                  <SelectTrigger id="titulacao">
                    <SelectValue placeholder="Selecione" />
                  </SelectTrigger>
                  <SelectContent>
                    {TITULACAO.map((option) => (
                      <SelectItem key={option} value={option}>
                        {option}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="area">Área de atuação</Label>
                <Select name="area" required>
                  <SelectTrigger id="area">
                    <SelectValue placeholder="Selecione" />
                  </SelectTrigger>
                  <SelectContent>
                    {AREAS.map((option) => (
                      <SelectItem key={option} value={option}>
                        {option}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="experiencia">Experiência prévia</Label>
                <Select name="experiencia" required>
                  <SelectTrigger id="experiencia">
                    <SelectValue placeholder="Selecione" />
                  </SelectTrigger>
                  <SelectContent>
                    {EXPERIENCIA.map((option) => (
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
              <h3 className="text-sm font-semibold">Parte 2 · Questionário de usabilidade</h3>
              <p className="text-sm text-muted-foreground">
                Para cada afirmação, marque de 1 (discordo totalmente) a 5 (concordo
                totalmente).
              </p>
            </div>

            {SUS_STATEMENTS.map((statement, index) => {
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
                          {SCALE_LABELS[score - 1] && (
                            <span className="ml-1 text-muted-foreground">
                              {SCALE_LABELS[score - 1]}
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
            <h3 className="text-sm font-semibold">Parte 3 · Interface e experiência</h3>

            {OPEN_QUESTIONS.map((question) => (
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

          <Button type="submit" disabled={sending} className="w-full">
            {sending ? 'Enviando…' : 'Enviar avaliação'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
