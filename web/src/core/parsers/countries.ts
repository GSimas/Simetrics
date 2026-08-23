/**
 * Extração de países a partir de strings de afiliação — ⇄ `extract_countries_robust`
 * (utils.py:2840).
 *
 * O campo de endereço do RIS é notoriamente sujo: mistura instituição, rua, CEP e
 * fragmentos de outras tags. Em vez de tentar entender a estrutura, o Python varre o
 * texto atrás de nomes geográficos conhecidos — abordagem mantida aqui.
 */

/** Dicionário geográfico do pipeline Python, incluindo as variantes do WoS. */
const COUNTRIES: readonly string[] = [
  'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina',
  'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados',
  'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina',
  'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia',
  'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 'Peoples R China',
  'Taiwan', 'Colombia', 'Comoros', 'Congo', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus',
  'Czech Republic', 'Czechoslovakia', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic',
  'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini',
  'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana',
  'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras',
  'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy',
  'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'North Korea', 'South Korea',
  'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya',
  'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives',
  'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia',
  'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',
  'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
  'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama',
  'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania',
  'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent', 'Samoa',
  'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles',
  'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia',
  'South Africa', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden',
  'Switzerland', 'Syria', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga',
  'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine',
  'United Arab Emirates', 'United Kingdom', 'UK', 'England', 'Scotland', 'Wales', 'North Ireland',
  'USA', 'United States', 'U S A', 'U.S.A.', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City',
  'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe',
];

/** Variantes que colapsam num nome canônico — ⇄ `MAPA_PAISES`. */
const COUNTRY_ALIASES: Readonly<Record<string, string>> = {
  'peoples r china': 'China',
  taiwan: 'Taiwan',
  usa: 'USA',
  'u s a': 'USA',
  'u.s.a.': 'USA',
  'united states': 'USA',
  uk: 'United Kingdom',
  england: 'United Kingdom',
  scotland: 'United Kingdom',
  wales: 'United Kingdom',
  'north ireland': 'United Kingdom',
};

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * O Python monta o padrão na ordem do dicionário, sem ordenar por comprimento. Como a
 * alternância do regex é gulosa da esquerda para a direita, essa ordem importa: manter o
 * array acima na ordem original é o que preserva a paridade nos casos ambíguos.
 */
const COUNTRY_PATTERN = new RegExp(`\\b(${COUNTRIES.map(escapeRegExp).join('|')})\\b`, 'gi');

/** Prefixos de tag RIS que vazam para dentro do campo de endereço. */
const RIS_NOISE = /\b(PU|C3|C1|AD|FU)\s+-\s+/g;

/** Nome canônico para uma ocorrência bruta. */
function canonical(match: string): string {
  const lower = match.toLowerCase();
  const alias = COUNTRY_ALIASES[lower];
  if (alias) return alias;
  // `.title()` do Python sobre o texto encontrado.
  return lower.replace(/\p{L}+/gu, (word) => word.charAt(0).toUpperCase() + word.slice(1));
}

/**
 * Países citados numa string de afiliação, únicos e ordenados, unidos por "; ".
 * Devolve `null` quando não encontra nenhum — o Python usa None e o normalizador
 * converte para texto vazio depois.
 */
export function extractCountries(affiliation: unknown): string | null {
  if (affiliation === null || affiliation === undefined) return null;

  const text = String(affiliation).trim();
  if (!text) return null;

  const cleaned = text.replace(RIS_NOISE, ' ');
  const found = new Set<string>();

  COUNTRY_PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = COUNTRY_PATTERN.exec(cleaned)) !== null) {
    found.add(canonical(match[1] as string));
  }

  return found.size > 0 ? [...found].sort().join('; ') : null;
}

/**
 * Variante reduzida usada pelo importador do PubMed — ⇄ `extrair_paises`
 * (utils.py:2109). O dicionário menor é intencional: reproduz o comportamento do Python,
 * que reconhece menos países nas afiliações do Medline.
 */
const PUBMED_COUNTRIES: readonly string[] = [
  'USA', 'United States', 'Canada', 'Brazil', 'China', 'Taiwan', 'United Kingdom', 'England',
  'Scotland', 'Wales', 'North Ireland', 'France', 'Germany', 'Italy', 'Spain', 'Portugal',
  'Netherlands', 'Switzerland', 'Sweden', 'Norway', 'Denmark', 'Finland', 'Belgium', 'Austria',
  'Russia', 'Australia', 'New Zealand', 'Japan', 'South Korea', 'India', 'South Africa', 'Mexico',
  'Argentina', 'Chile', 'Colombia', 'Peru', 'Israel', 'Saudi Arabia', 'Iran', 'Turkey', 'Egypt',
];

const PUBMED_ALIASES: Readonly<Record<string, string>> = {
  usa: 'USA',
  'united states': 'USA',
  uk: 'United Kingdom',
  england: 'United Kingdom',
};

const PUBMED_PATTERN = new RegExp(`\\b(${PUBMED_COUNTRIES.map(escapeRegExp).join('|')})\\b`, 'gi');

export function extractCountriesPubmed(affiliations: readonly string[]): string | null {
  if (affiliations.length === 0) return null;

  const text = affiliations.join(' ');
  const found = new Set<string>();

  PUBMED_PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = PUBMED_PATTERN.exec(text)) !== null) {
    const lower = (match[1] as string).toLowerCase();
    found.add(
      PUBMED_ALIASES[lower] ??
        lower.replace(/\p{L}+/gu, (word) => word.charAt(0).toUpperCase() + word.slice(1)),
    );
  }

  return found.size > 0 ? [...found].sort().join('; ') : null;
}
