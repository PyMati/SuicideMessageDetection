from detector.preprocessor import Preprocessor
from detector.detector import Detector


def main():
    str_ = "It was just very boring that and my friends was just not respectable for me"
    prcs = Preprocessor()
    detector = Detector(prcs)
    print(detector.predict(str_))


if __name__ == "__main__":
    main()
