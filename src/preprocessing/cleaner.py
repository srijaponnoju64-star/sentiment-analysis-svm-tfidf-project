"""
Text cleaning and normalization.
Works standalone (regex + built-in stopword list) and automatically uses
NLTK's tokenizer/lemmatizer/stopwords if they're installed & downloaded.
"""
import re

_HAS_NLTK = False
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    try:
        nltk_stopwords.words("english")
        word_tokenize("test")
        _lemmatizer = WordNetLemmatizer()
        _lemmatizer.lemmatize("running")
        _HAS_NLTK = True
    except LookupError:
        # NLTK is installed but data hasn't been downloaded yet
        _HAS_NLTK = False
except ImportError:
    _HAS_NLTK = False

# Fallback stopword list (used until `nltk.download(...)` has been run)
_FALLBACK_STOPWORDS = set("""
a an the and or but if while is are was were be been being have has had
do does did will would shall should may might must can could of at by
for with about against between into through during before after above
below to from up down in out on off over under again further then once
here there all any both each few more most other some such no nor not
only own same so than too very s t just don now i me my myself we our
ours ourselves you your yours yourself yourselves he him his himself
she her hers herself it its itself they them their theirs themselves
what which who whom this that these those am
""".split())

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")

_CONTRACTIONS = {
    "don't": "do not", "won't": "will not", "can't": "cannot",
    "n't": " not", "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am",
}


def expand_contractions(text: str) -> str:
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def basic_clean(text: str) -> str:
    """URL/mention/HTML removal, contraction expansion, lowercasing."""
    text = str(text)
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)   # keep the word, drop the '#'
    text = text.lower()
    text = expand_contractions(text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str):
    if _HAS_NLTK:
        return word_tokenize(text)
    return text.split()


def remove_stopwords(tokens):
    stopword_set = set(nltk_stopwords.words("english")) if _HAS_NLTK else _FALLBACK_STOPWORDS
    return [t for t in tokens if t not in stopword_set and len(t) > 1]


def lemmatize(tokens):
    if _HAS_NLTK:
        return [_lemmatizer.lemmatize(t) for t in tokens]
    return tokens  # no-op fallback


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize -> join."""
    cleaned = basic_clean(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_batch(texts):
    return [preprocess(t) for t in texts]


if __name__ == "__main__":
    samples = [
        "I LOVED this product!!! Check it out: https://example.com @friend #amazing",
        "This is terrible, I can't believe it broke so fast :(",
    ]
    for s in samples:
        print(f"IN:  {s}")
        print(f"OUT: {preprocess(s)}\n")
    print(f"NLTK available: {_HAS_NLTK}")