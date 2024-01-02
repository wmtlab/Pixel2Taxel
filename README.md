# Toward Spatial Temporal Consistency of Joint Visual Tactile Perception in VR Applications

## Introduction
We propose a new data collection system, which scans the visual and tactile information across various textures simultaneously and establish a spatiotemporal mapping between visual and tactile signals based on coordinates. The developed scheme enables the creation of a spatially consistent visual-tactile datasets. You can find more details in the [paper](link).


## Result_Data
We have uploaded the processed data, which can be found in [Result_Data\Data](./Result_Data/Data). The file structure is as:

```bash
Result_Data
+-- Data
|  +-- object1  # the first object.
|  |  +-- object1.png 
|  |  +-- object1_acc.png #Image of the acceleration. 
|  +-- other objects
```

## Usage for V-Touching
In the [V-Touching](https://github.com/wmtlab/V-Touching), as mentioned in our paper, we can load this **Result_Data** to experience this VR application.
Therefore, we have uploaded the [usage of Result_Data](./Result_Data/README.md) in V-Touching.

## Raw Data
We provide the [download link](https://drive.google.com/file/d/1dyjluW00Z5RIfV7Saul47CdEo3FqGSM0/view?usp=drive_link) for the raw data, and its structure is as follows:
```bash
Raw_Data
+-- 1
|  +-- sensor_data # Raw data collected
|  |  +-- ...
|  +-- img_xxx.png  # Raw image
|  +-- postions_1.csv # Robot coordinates for calibration
|  +-- select_points_xxx.txt # Pixel coordinates for calibration
|  +-- transform_to_crop.txt # Cropp region
+-- other indices (1, 2, 3, ...)
+-- name.json # Index to name mapping
+-- PerspectiveTransform1/2.npy # Perspective transformation matrices
```

## Process_Code
1. We have uploaded the [code](Process_Code)  for processing the raw data.
If you want to run our code, please install the required dependencies first by executing the command:
    ```bash
    pip install -r Process_Code\requirements.txt
    ```
2. Next, you can download the [raw data](https://drive.google.com/file/d/1dyjluW00Z5RIfV7Saul47CdEo3FqGSM0/view?usp=drive_link) and extract it so that the Raw_Data folder is at the same level as the Process_Code folder. Then, you can execute the command in the terminal:
    ```python
    python Process_Code\run.py --root=Raw_Data --index=1 
    ```
    Please note that you need to specify the following two parameters:
    ```
    --root: Specifies the root directory path of the raw data
    --index: Specifies the data index to be processed
    ```
3. If successful, the terminal will print:
    ```
    Succeed! The result of warpPerspective is in the folder:  Raw_Data\1\out\transform
    Succeed! The result of crop is in the folder:  Raw_Data\1\out\transform
    Succeed! The final result is in the folder:  Raw_Data\1\out\smooth_ceramic_tile
    ```

## Citation

```
@article{zhao2023toward,
  title={Toward Spatial Temporal Consistency of Joint Visual Tactile Perception in VR Applications},
  author={Zhao, Fuqiang and Zhang, Kehan and Liu, Qian and Lyu, Zhuoyi},
  journal={arXiv preprint arXiv:2312.16391},
  year={2023}
}
```


