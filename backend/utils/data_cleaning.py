# data_cleaning.py
# -*- coding: utf-8 -*-

import re
import string
import emoji
import pyarabic.araby as ar
import Stemmer

# Initialize the Arabic stemmer once
_stemmer = Stemmer.Stemmer('arabic')

def clean_text(text: str) -> str:
    """
    Clean and normalize Arabic text:
      1. Remove URLs
      2. Collapse whitespace
      3. Remove digits
      4. Strip diacritics (tashkeel) and tatweel
      5. Remove hashtags, mentions, underscores
      6. Remove punctuation
      7. Convert emojis to text
      8. Collapse repeated characters
      9. Stem each word
     10. Normalize hamza and alef variants
    """
    # 1. Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # 2. Collapse whitespace and strip
    text = re.sub(r'\s+', ' ', text).strip()
    # 3. Remove digits
    text = re.sub(r'\d+', ' ', text)
    # 4. Strip diacritics and tatweel
    text = ar.strip_tashkeel(text)
    text = ar.strip_tatweel(text)
    # 5. Remove common symbols
    for ch in ('#', '@', '_'):
        text = text.replace(ch, ' ')
    # 6. Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # 7. Demojize (convert emojis to text)
    text = emoji.demojize(text, delimiters=(' ', ' '))
    # 8. Collapse repeated characters (e.g., cooool → col)
    text = re.sub(r'(.)\1+', r'\1', text)
    # 9. Stem each word
    tokens = text.split()
    tokens = [_stemmer.stemWord(tok) for tok in tokens]
    text = ' '.join(tokens)
    # 10. Normalize hamza/alef variants
    text = (text.replace('آ', 'ا')
                .replace('إ', 'ا')
                .replace('أ', 'ا')
                .replace('ؤ', 'و')
                .replace('ئ', 'ي'))
    return text