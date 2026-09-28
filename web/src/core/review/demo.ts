import { defaultQualityAnswers } from './quality';
import type { ScreeningRecord } from './records';
import { createReview } from './state';
import type { RecordScreening, ReviewState, StudyExtraction } from './types';

/**
 * Revisão de amostra que acompanha a base de exemplo (973 registros sobre memética): uma
 * revisão de escopo com protocolo preenchido, triagem em andamento, avaliação de qualidade
 * e extração de parte dos estudos. Serve para o exemplo e o tour guiado mostrarem o módulo
 * com conteúdo real, só para visualização; a cópia editável a leva junto.
 *
 * Tudo é determinístico: os registros são escolhidos pelo título, na ordem da base, e os
 * identificadores são fixos — o mesmo exemplo gera sempre a mesma revisão.
 */

const TEXT = {
  pt: {
    title: 'Memética e evolução cultural: revisão de escopo',
    objective:
      'Mapear como a literatura científica usa o conceito de meme como unidade de transmissão cultural: áreas, tipos de estudo e modelos empregados.',
    population: 'Estudos que aplicam a teoria memética',
    concept: 'Memes como unidades de transmissão e evolução cultural',
    context: 'Qualquer área do conhecimento, sem recorte de período',
    questions: [
      'Em que áreas do conhecimento a memética é aplicada?',
      'Que tipos de estudo e de modelo sustentam as aplicações?',
    ],
    concepts: [
      { label: 'Memética', terms: ['memetic*', 'meme*'] },
      { label: 'Evolução cultural', terms: ['cultural evolution', 'cultural transmission', 'information diffusion'] },
    ],
    inclusion: ['Trata memes como unidades de transmissão cultural', 'Artigo, capítulo ou trabalho em evento'],
    exclusion: ['Usa "meme" só como humor de internet, sem teoria', 'Fora do tema da revisão', 'Texto completo indisponível'],
    quality: [
      'O objetivo do estudo está claramente descrito?',
      'O conceito de meme é definido explicitamente?',
      'O método é adequado ao objetivo?',
    ],
    fields: {
      area: 'Área',
      areas: ['Biologia', 'Linguística', 'Computação', 'Ciências sociais'],
      design: 'Tipo de estudo',
      designs: ['Teórico', 'Empírico', 'Simulação'],
      model: 'Usa modelo computacional',
    },
  },
  en: {
    title: 'Memetics and cultural evolution: a scoping review',
    objective:
      'Map how the scientific literature uses the meme concept as a unit of cultural transmission: fields, study types and models employed.',
    population: 'Studies applying memetic theory',
    concept: 'Memes as units of cultural transmission and evolution',
    context: 'Any field of knowledge, no period restriction',
    questions: ['In which fields of knowledge is memetics applied?', 'What study types and models support the applications?'],
    concepts: [
      { label: 'Memetics', terms: ['memetic*', 'meme*'] },
      { label: 'Cultural evolution', terms: ['cultural evolution', 'cultural transmission', 'information diffusion'] },
    ],
    inclusion: ['Treats memes as units of cultural transmission', 'Journal article, chapter or conference paper'],
    exclusion: ['Uses "meme" only as internet humor, with no theory', 'Off-topic for the review', 'Full text unavailable'],
    quality: ['Is the study aim clearly stated?', 'Is the meme concept explicitly defined?', 'Is the method suited to the aim?'],
    fields: {
      area: 'Field',
      areas: ['Biology', 'Linguistics', 'Computing', 'Social sciences'],
      design: 'Study type',
      designs: ['Theoretical', 'Empirical', 'Simulation'],
      model: 'Uses a computational model',
    },
  },
};

const AT = '2026-01-01T00:00:00.000Z';

function areaOf(title: string): number {
  if (/algorithm|comput|network|agent|optimi|evolutionary|robot|software/i.test(title)) return 2;
  if (/song|bird|gene|biolog|animal|species/i.test(title)) return 0;
  if (/language|translat|linguist|word|discourse/i.test(title)) return 1;
  return 3;
}

export function buildDemoReview(records: readonly ScreeningRecord[], locale: 'pt' | 'en', reviewerId: string): ReviewState {
  const text = TEXT[locale];
  const review = createReview(reviewerId, 'scoping');
  const id = (prefix: string, index: number) => `demo-${prefix}-${index + 1}`;

  review.title = text.title;
  review.objective = text.objective;
  review.frameworkValues = { population: text.population, concept: text.concept, context: text.context };
  review.questions = text.questions.map((question, index) => ({ id: id('q', index), text: question }));
  review.concepts = text.concepts.map((concept, index) => ({ id: id('c', index), label: concept.label, terms: [...concept.terms] }));
  review.searchTargets = ['generic', 'scopus', 'wos'];
  review.criteria = [
    ...text.inclusion.map((criterion, index) => ({ id: id('inc', index), kind: 'inclusion' as const, text: criterion })),
    ...text.exclusion.map((criterion, index) => ({ id: id('exc', index), kind: 'exclusion' as const, text: criterion })),
  ];
  const [humor, offTopic] = text.exclusion.map((_, index) => id('exc', index)) as [string, string, string];

  // Triagem em andamento: parte decidida, o resto pendente — como uma revisão real no meio.
  const decisions: Record<string, RecordScreening> = {};
  const memetic = records.filter((record) => /memetic/i.test(record.title)).slice(0, 14);
  const humorous = records.filter((record) => /internet meme|humou?r|funny|joke|viral/i.test(record.title)).slice(0, 5);
  const others = records.filter((record) => !/meme/i.test(record.title)).slice(0, 12);

  memetic.forEach((record, index) => {
    const screening: RecordScreening = { ta: index % 5 === 4 ? 'maybe' : 'include', updatedAt: AT };
    if (index < 9) screening.ft = 'include';
    else if (index < 11) Object.assign(screening, { ft: 'exclude', ftReason: offTopic });
    else if (index === 11) screening.ft = 'not-retrieved';
    else if (index === 12) Object.assign(screening, { ft: 'exclude', ftReason: humor });
    // O último fica pendente no texto completo.
    decisions[record.key] = screening;
  });
  humorous.forEach((record) => {
    decisions[record.key] ??= { ta: 'exclude', taReason: humor, updatedAt: AT };
  });
  others.forEach((record) => {
    decisions[record.key] ??= { ta: 'exclude', taReason: offTopic, updatedAt: AT };
  });
  review.decisions = decisions;

  // Qualidade: três perguntas, respostas padrão e nota de corte, sem excluir (é escopo).
  review.qualityQuestions = text.quality.map((question, index) => ({ id: id('qq', index), text: question }));
  review.qualityAnswers = defaultQualityAnswers(locale).map((answer, index) => ({ ...answer, id: id('qa', index) }));
  review.qualityCutoff = 2;
  const [yes, partial, no] = review.qualityAnswers.map((answer) => answer.id) as [string, string, string];
  const patterns = [
    [yes, yes, yes],
    [yes, partial, yes],
    [partial, no, partial],
    [yes, yes, partial],
    [yes, partial, partial],
    [partial, partial, no],
  ];
  const included = memetic.slice(0, 9);
  included.slice(0, patterns.length).forEach((record, index) => {
    const answers = patterns[index]!;
    review.quality[record.key] = Object.fromEntries(review.qualityQuestions.map((question, q) => [question.id, answers[q]!]));
  });

  // Extração: três campos, cinco estudos concluídos e um em andamento.
  const fields = text.fields;
  review.extractionFields = [
    { id: 'demo-f-area', label: fields.area, type: 'select', options: [...fields.areas] },
    { id: 'demo-f-design', label: fields.design, type: 'select', options: [...fields.designs] },
    { id: 'demo-f-model', label: fields.model, type: 'boolean', options: [] },
  ];
  included.slice(0, 6).forEach((record, index) => {
    const area = areaOf(record.title);
    const extraction: StudyExtraction = {
      values: {
        'demo-f-area': fields.areas[area]!,
        'demo-f-design': fields.designs[area === 2 ? 2 : index % 2]!,
        'demo-f-model': area === 2,
      },
      done: index < 5,
    };
    review.extraction[record.key] = extraction;
  });

  return review;
}
