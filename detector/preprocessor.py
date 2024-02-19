import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(
        self,
        model_path: str = "svc.pickle",
        vectorizer_path: str = "vectorizer.pickle",
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = word_tokenize

        self.model = self.load_object(model_path, "Detector")
        self.vectorizer = self.load_object(vectorizer_path, "Vectorizer")

    def load_object(self, path: str, type: str):
        try:
            with open(path, "rb") as file:
                print(f"Loading from {path} object of type {type}")
                return pickle.load(file)
        except Exception as e:
            print("An exception occured while reading file...")
            print(e)

    def get_vectorizer(self):
        return self.vectorizer

    def get_model(self):
        return self.model

    def preprocess_input(self, input: str):
        input_tokens = self.tokenizer(input)
        preprocessed_tokens = [
            self.lemmatizer.lemmatize(word.lower())
            for word in input_tokens
            if word.lower() not in self.stopwords and word.isalpha()
        ]
        preprocessed_text = " ".join(preprocessed_tokens)
        input_transformed = self.get_vectorizer().transform([preprocessed_text])
        return input_transformed
