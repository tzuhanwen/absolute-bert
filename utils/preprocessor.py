from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

stopwords = stopwords.words("english")


class StopwordsPreprocessor:

    regex_tokenizer = RegexpTokenizer(r"(?<=\s)?[a-zA-Z]{1,}(?=\s)?")
    # tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')

    @classmethod
    def __call__(cls, text) -> str:
        return " ".join(
            [word for word in cls.regex_tokenizer.tokenize(text) if word.lower() not in stopwords]
        )
