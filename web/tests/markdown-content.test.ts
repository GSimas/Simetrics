import { describe, expect, it } from 'vitest';

import { MarkdownContent } from '@/components/MarkdownContent';

// Resposta típica da Simi; durante o streaming o renderizador vê cada prefixo dela.
const RESPONSE = `## Trabalhos do Brasil

A base tem **42 documentos** do Brasil.

### Mais citados
1. *Ecologia do conhecimento* — 120 citações
2. \`Gestão\` — 80 citações

- Autor A
- Autor B

| País | Docs |
|:-----|-----:|
| Brasil | 42 |

> Nota final
---`;

describe('MarkdownContent', () => {
  it('termina em todo prefixo de uma resposta em streaming', () => {
    for (let end = 1; end <= RESPONSE.length; end++) {
      MarkdownContent({ content: RESPONSE.slice(0, end) });
    }
  });

  it.each(['##', 'texto\n##', '- ', '1. ', '  # recuado', '#####  cinco', '* '])(
    'termina com marca incompleta %j',
    (content) => {
      expect(MarkdownContent({ content })).not.toBeNull();
    },
  );
});
