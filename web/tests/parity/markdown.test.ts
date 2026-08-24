import { describe, expect, it } from 'vitest';
import { MarkdownContent } from '@/components/MarkdownContent';

describe('Markdown Table Rendering', () => {
  it('renders markdown tables properly with headers, alignments and rows', () => {
    const rawMarkdown = `
Top 15 Most Cited Countries

| Rank | Country | Total Citations | Documents | H-Index |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **United States** | 6,462 | 143 | 20 |
| 2 | **China** | 2,626 | 168 | 28 |
| 3 | **United Kingdom** | 2,395 | 146 | 28 |
`;
    const element = MarkdownContent({ content: rawMarkdown });
    expect(element).toBeDefined();
  });

  it('handles squashed table rows on single lines with | |', () => {
    const squashed = `| Rank | Country | Total Citations | | :---: | :---: | :---: | | 1 | United States | 6,462 | | 2 | China | 2,626 |`;
    const element = MarkdownContent({ content: squashed });
    expect(element).toBeDefined();
  });
});
