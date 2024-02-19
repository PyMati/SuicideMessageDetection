from detector.preprocessor import Preprocessor
from detector.detector import Detector


def main():
    message: str
    prcs = Preprocessor()
    detector = Detector(prcs)
    print("Paste message which you got.")
    while True:
        message = input("")
        if message == "exit":
            break
        print(detector.predict(message))


if __name__ == "__main__":
    main()
