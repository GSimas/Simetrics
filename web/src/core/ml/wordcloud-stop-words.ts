/**
 * Stop words da biblioteca `wordcloud` do Python (192 termos).
 *
 * NÃO é a mesma lista do scikit-learn usada no TF-IDF — são bibliotecas diferentes, com
 * listas diferentes, e o pipeline Python usa cada uma no seu lugar. Copiada literalmente
 * para que a nuvem de palavras produza exatamente os mesmos termos.
 *
 * Os termos vão entre aspas duplas porque vários contêm apóstrofo ("don't", "you're").
 */
export const WORDCLOUD_STOP_WORDS: ReadonlySet<string> = new Set([
  "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any",
  "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between",
  "both", "but", "by", "can", "can't", "cannot", "com", "could", "couldn't", "did", "didn't",
  "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "else", "ever", "few",
  "for", "from", "further", "get", "had", "hadn't", "has", "hasn't", "have", "haven't",
  "having", "he", "he'd", "he'll", "he's", "hence", "her", "here", "here's", "hers", "herself",
  "him", "himself", "his", "how", "how's", "however", "http", "i", "i'd", "i'll", "i'm", "i've",
  "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "just", "k", "let's",
  "like", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
  "on", "once", "only", "or", "other", "otherwise", "ought", "our", "ours", "ourselves", "out",
  "over", "own", "r", "same", "shall", "shan't", "she", "she'd", "she'll", "she's", "should",
  "shouldn't", "since", "so", "some", "such", "than", "that", "that's", "the", "their",
  "theirs", "them", "themselves", "then", "there", "there's", "therefore", "these", "they",
  "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
  "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
  "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
  "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "www", "you", "you'd",
  "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
]);

/** Termos acadêmicos genéricos que o Simetrics adiciona à lista — ⇄ utils.py:2689. */
export const SIMETRICS_EXTRA_STOP_WORDS: readonly string[] = [
  "research", "study", "analysis", "results", "using", "paper", "article", "author", "will",
  "may", "can",
];
