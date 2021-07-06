# em-calibration
Electron microscopy image calibration utilities for grids and measurements

## Usage
Use python 3.6 or later

## GridProcessor Usage
1. Initialize the GridProcessor:
Grid Processor accept two parameters: 1. the path of the image, 2. size of each grid(in nanomaters)

```
path = 'C:\\Users\\08_01615.tif'
processor = GridProcessor(path, '463nm')
```



2. Calculate the average distance among grid lines (nanometers per pixel)
```
path = 'C:\\Users\\08_01615.tif'
processor = GridProcessor(path, '463nm')
```

## GridIdentifier Usage
This is a fast identifier to help you quickly identify if an image is a grid image is not. 
For example:
```
from grid_identifier import is_grid
path = 'C:\\Users\\08_01615.png'
print(if_grid(path))
```
