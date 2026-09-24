/**
 * Continente de cada país do dicionário geográfico (`parsers/countries.ts`), com as
 * variantes que as bases usam (WoS: "Peoples R China", "U S A"...). América Central e
 * Caribe entram na América do Norte; Oriente Médio, Turquia e Cáucaso, na Ásia; Rússia,
 * na Europa.
 */
export const CONTINENTS = [
  'Africa',
  'North America',
  'South America',
  'Asia',
  'Europe',
  'Oceania',
] as const;

export type Continent = (typeof CONTINENTS)[number];

const COUNTRIES_BY_CONTINENT: Record<Continent, readonly string[]> = {
  Africa: [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cape Verde',
    'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo',
    'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea',
    'Eswatini', 'Swaziland', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau',
    'Ivory Coast', "Cote d'Ivoire", 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi',
    'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria',
    'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia',
    'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia',
    'Zimbabwe',
  ],
  'North America': [
    'Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba',
    'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras',
    'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Puerto Rico', 'Saint Kitts and Nevis',
    'Saint Lucia', 'Saint Vincent', 'Trinidad and Tobago', 'USA', 'U S A', 'U.S.A.',
    'United States', 'United States of America',
  ],
  'South America': [
    'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru',
    'Suriname', 'Uruguay', 'Venezuela',
  ],
  Asia: [
    'Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia',
    'China', 'Peoples R China', 'Georgia', 'Hong Kong', 'India', 'Indonesia', 'Iran', 'Iraq',
    'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Macau',
    'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'North Korea', 'South Korea', 'Korea',
    'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia', 'Singapore',
    'Sri Lanka', 'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkey', 'Turkiye',
    'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen',
  ],
  Europe: [
    'Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria',
    'Croatia', 'Cyprus', 'Czech Republic', 'Czechia', 'Czechoslovakia', 'Denmark', 'Estonia',
    'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kosovo',
    'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco',
    'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania',
    'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland',
    'Ukraine', 'United Kingdom', 'UK', 'England', 'Scotland', 'Wales', 'North Ireland',
    'Northern Ireland', 'Vatican City',
  ],
  Oceania: [
    'Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand',
    'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu',
  ],
};

const CONTINENT_BY_COUNTRY = new Map<string, Continent>(
  CONTINENTS.flatMap((continent) =>
    COUNTRIES_BY_CONTINENT[continent].map((country) => [country.toLowerCase(), continent] as const),
  ),
);

/** Continente do país, sem diferenciar maiúsculas ("Usa", "USA"); `null` se desconhecido. */
export function continentOf(country: string): Continent | null {
  return CONTINENT_BY_COUNTRY.get(country.trim().toLowerCase()) ?? null;
}
