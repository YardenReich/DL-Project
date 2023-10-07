## Workshop On Deep Learning - Climate Video

This is our project in a workshop on deep learning.


## Table of Contents

- [Installation](#installation)
- [Running examples](#running-examples)
- [Collab Notebooks](#collab-notebooks)


## Installation

To get started with this project, follow these steps:

1. Clone the repository: `git clone https://github.com/YardenReich/DL-Project.git`
2. Install dependencies: `pip install requirements.txt`

The models should be downloaded when you run the files, if not you can run:
```
python Download_models.py
```
You can also find them in:
- [First Model](https://drive.google.com/uc?export=download&id=1uNS2G3908KpiJ7sPfLYev36NTGJHuptM)
- [Second Model](https://drive.google.com/uc?export=download&id=1SZYE7CDWMJehD0MN0c8inp9e-IZKRHq1)

## Running examples:
To run interpolation:
```
python interpolation.py
```
To run attribute manipulation:
```
python interpolation.py --attribute
```
To add your own images to interpolation:
```
python interpolation.py --image1-path {image path} --image2-path {image path}
```
To run clip:
```
python clip_video.py
```
To add your own images to clip:
```
python clip_video.py --image-path {image path}
```
To see more options:
```
python interpolation.py -h
```
```
python clip_video.py -h
```

## Collab Notebooks

- [Interpolation](https://colab.research.google.com/drive/1wfDLjRhVif3-WKAIvzMW4eCTyUQ6lOg2?usp=sharing)
- [CLIP](https://colab.research.google.com/drive/1euZJbCql8QaGQdl2gGAhdOpVgwoq8Hg1?usp=sharing)

