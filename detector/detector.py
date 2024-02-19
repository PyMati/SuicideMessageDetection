from .preprocessor import Preprocessor


class Detector:
    def __init__(self, preprocessor_instance: Preprocessor):
        self.preprocessor = preprocessor_instance

        self.model = self.preprocessor.get_model()

    def predict(self, inp):
        preprocessed_input = self.preprocessor.preprocess_input(inp)
        prediction = self.model.predict(preprocessed_input)
        if prediction[0] == 0:
            return "Suicide alert"

        return "Sadness"
