## Workshop On Deep Learning - Climate Video

This is our project in a workshop on deep learning.

The model checkpoints can be downloaded from:
- [First Model](https://drive.google.com/uc?export=download&id=1uNS2G3908KpiJ7sPfLYev36NTGJHuptM)
- [Second Model](https://drive.google.com/uc?export=download&id=1SZYE7CDWMJehD0MN0c8inp9e-IZKRHq1)


## Table of Contents

- [Installation](#installation)
- [Running examples](#running-examples)
- [Collab Notebooks](#collab-notebooks)


## Installation

To get started with this project, follow these steps:

1. Clone the repository: `git clone https://github.com/YardenReich/DL-Project.git`
2. Install dependencies: `pip install requirements.txt`

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

- [Interpolation](https://github.com/YardenReich/DL-Project/blob/main/collab%20notebooks/interpolation.ipynb)
- [CLIP](https://github.com/YardenReich/DL-Project/blob/main/collab%20notebooks/VQGAN_and_CLIP.ipynb)

