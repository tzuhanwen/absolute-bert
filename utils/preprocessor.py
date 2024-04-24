from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

class Stopwords_preprocessor:
    
    regex_tokenizer = RegexpTokenizer(r'(?<=\s)?[a-zA-Z]{1,}(?=\s)?')
    # tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
    
    @staticmethod
    def __call__(text):
        return ' '.join([word for word in Stopwords_preprocessor.regex_tokenizer.tokenize(text) if word.lower() not in stopwords])