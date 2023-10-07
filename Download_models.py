import gdown
import os


def download_first_model():
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists("checkpoints/model_first.pt"):
        print("Downloading the first model")
        url = "https://drive.google.com/uc?export=download&id=1uNS2G3908KpiJ7sPfLYev36NTGJHuptM"
        output = "checkpoints/model_first.pt"
        gdown.download(url, output, quiet=True)
        print("Done")


def download_second_model():
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists("checkpoints/model_second.pt"):
        print("Downloading the second model")
        url = "https://drive.google.com/uc?export=download&id=1SZYE7CDWMJehD0MN0c8inp9e-IZKRHq1"
        output = "checkpoints/model_second.pt"
        gdown.download(url, output, quiet=True)
        print("Done")


if __name__ == "__main__":
    download_first_model()
    download_second_model()
