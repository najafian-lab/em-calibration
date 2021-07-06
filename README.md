# em-calibration
Electron microscopy image calibration utilities for grids and measurements

## Usage
Use python 3.6 or later

## GridProcessir Usage
1. Initialize the GridProcessor:

```
path = 'C:\\Users\\08_01615.tif'
processor = GridProcessor(path, '463nm')
```
Grid Processor accept two parameters: 1. the path of the image, 2. size of each grid(in nanomaters)


2. Calculate the average distance among grid lines
```
path = 'C:\\Users\\08_01615.tif'
processor = GridProcessor(path, '463nm')
```