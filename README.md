# SubPipe-dataset :diving_mask:
A Submarine Pipeline Inspection Dataset for Segmentation and Visual-inertial Localization.
Here, you will find all the relevant information regarding the dataset introduced in [our paper:](https://arxiv.org/abs/2401.17907)
> Álvarez-Tuñón, O., Marnet, L. R., Antal, L., Aubard, M., Costa, M., & Brodskiy, Y. (2024). SubPipe: A Submarine Pipeline Inspection Dataset for Segmentation and Visual-inertial Localization. arXiv preprint arXiv:2401.17907.

If any of this work has been useful in your research, please consider citing us :smiley:

![](https://raw.githubusercontent.com/remaro-network/SubPipe-dataset/main/media/lauv-paper.png)

- [SubPipe-dataset :diving\_mask:](#subpipe-dataset-diving_mask)
  - [1. The dataset](#1-the-dataset)
    - [1.1 Dataset link](#11-dataset-link)
    - [1.2 The dataset structure](#12-the-dataset-structure)
    - [1.3 Camera parameters](#13-camera-parameters)
      - [Cam0 - GoPro Hero 10](#cam0---gopro-hero-10)
  - [1.3 Dataset metrics on SubPipe](#13-dataset-metrics-on-subpipe)
- [2. The experiments](#2-the-experiments)
  - [2.1 SLAM](#21-slam)
  - [2.2 RGB segmentation](#22-rgb-segmentation)
  - [2.3 Object detection on Side-scan sonar images](#23-object-detection-on-side-scan-sonar-images)
- [Acknowledgements](#acknowledgements)


## 1. The dataset

 ### 1.1 Dataset link
 You can download SubPipe from [the following link](https://zenodo.org/doi/10.5281/zenodo.10053564)

###  1.2 The dataset structure
The dataset is divided into 5 Chunks. We exemplify SubPipe's structure with Chunk 0 as:

```
SubPipe
├── config.yaml
├── DATA
│   ├──Chunk0
│   │   ├── Acceleration.csv
│   │   ├── Altitude.csv
│   │   ├── AngularVelocity.csv
│   │   ├── Cam0_images
│   │   │   ├── <timestamp0>.jpg
│   │   │   ├── ...
│   │   │   └── <timestampN>.jpg
│   │   ├── Cam1_images
│   │   │   ├── <timestamp0>.jpg
│   │   │   ├── ...
│   │   │   └── <timestampN>.jpg
│   │   ├── Depth.csv
│   │   ├── EstimatedState.csv
│   │   ├── ForwardDistance.csv
│   │   ├── Pressure.csv
│   │   ├── Rpm.csv
│   │   ├── Segmentation
│   │   │   ├── <timestamp0>.png
│   │   │   ├── <timestamp0>_label.png
│   │   │   ├── ...
│   │   │   ├── <timestampN>.png
│   │   │   └── <timestampN>_label.png
│   │   ├── SSS_HF_images
│   │   │   ├── COCO_Annotation
│   │   │   |   └── coco_format.json 
│   │   │   ├── Image
│   │   │   |   ├── <timestamp0>.pbm
│   │   │   |   ├── ...
│   │   │   |   └── <timestampN>.pbm
│   │   │   ├── YOLO_Annotation
│   │   │   |   ├── <timestamp0>.txt
│   │   │   |   ├── ...
│   │   │   |   └── <timestampN>.txt
│   │   ├── SSS_LF_images
│   │   │   ├── COCO_Annotation
│   │   │   |   └── coco_format.json 
│   │   │   ├── Image
│   │   │   |   ├── <timestamp0>.pbm
│   │   │   |   ├── ...
│   │   │   |   └── <timestampN>.pbm
│   │   │   └── YOLO_Annotation
│   │   │       ├── <timestamp0>.txt
│   │   │       ├── ...
│   │   │       └── <timestampN>.txt
│   │   ├── Pressure.csv
│   │   ├── Temperature.csv
│   │   └── WaterVelocity.csv
│   ├──Chunk1
│   ├──Chunk2
│   ├──Chunk3
│   └──Chunk4
```

###  1.3 Camera parameters

####  Cam0 - GoPro Hero 10
> - Resolution: 1520x2704
> - fx = 1612.36
> - fy = 1622.56
> - cx = 1365.43
> - cy = 741.27
> - k1,k2, p1, p2 = [−0.247, 0.0869, −0.006, 0.001]

## 1.3 Dataset metrics on SubPipe
Our paper proposes a set of metrics to compare SubPipe with existing datasets.
The folder `dataset_metrics` within this repo includes:
- The results from applying the metrics on state-of-the-art datasets -> :open_file_folder: [here](https://github.com/remaro-network/SubPipe-dataset/tree/main/dataset_metrics/results)
- A notebook to plot the metrics -> :notebook: [here](https://github.com/remaro-network/SubPipe-dataset/blob/main/dataset_metrics/plot_metrics.ipynb)
- The code for deploying the metrics -> [here](https://github.com/remaro-network/SubPipe-dataset/blob/main/dataset_metrics/dataset_metrics.py). Note: you will need to configure your own dataloader.

The metrics proposed are delentropy and motion diversity. For more info about those metrics, consider reading [our paper](https://arxiv.org/abs/2401.17907).

# 2. The experiments
Our paper proposes a comprehensive set of experiments to demonstrate SubPipe's performance on different use-cases.

## 2.1 visual SLAM
The algorithms tested in this paper are the geometry-based ORB-SLAM3 and DSO and the learning-based algorithm TartanVO.
The data loaders for ORB-SLAM and DSO are gathered together in [this repository](https://github.com/olayasturias/monocular_visual_slam_survey) that surveys monocular visual SLAM algorithms. It includes [scripts](https://github.com/olayasturias/monocular_visual_slam_survey/tree/main/scripts) for running the algorithms in your favourite datasets, including Subpipe. For more detailed info, we recommend you to read the repo's README.

## 2.2 RGB segmentation
- [Segformer](https://github.com/FrancescoSaverioZuppichini/SegFormer)

## 2.3 Object detection on Side-scan sonar images
Each sonar image was created after 20 “ping” (after every 20 new lines) which corresponds to approx. ~1 image / second.

Regarding the object detection annotations, we provide both COCO and YOLO formats for each annotation. A single COCO annotation file is provided per each chunk and per each frequency (low frequency vs. high frequency), whereas the YOLO annotations are provided for each SSS image file.

Metadata about the side-scan sonar images contained in this dataset:

Side-scan Sonar Images:
- \# Low Frequency (LF): 5000
- LF image size: 2500 x 500
- \# High Frequency (HF): 5030
- HF image size: 5000 x 500
- Total number of images: 10030

Number of Annotations:
- \# Low Frequency: 3163
- \# High Frequency: 3172 
- Total number of annotations: 6335

# Acknowledgements

<strong>SubPipe</strong> is a public dataset of a submarine outfall pipeline, property of Oceanscan-MST. This dataset was acquired with a Light Autonomous Underwater Vehicle by Oceanscan-MST, within the scope of Challenge Camp 1 of H2020 [REMARO](https://remaro.eu/) project.

More information about Oceanscan-MST can be found at this [link](https://www.oceanscan-mst.com/).

<a href="https://remaro.eu/">
    <img height="60" alt="REMARO Logo" src="https://remaro.eu/wp-content/uploads/2020/09/remaro1-right-1024.png">
</a>
<a href="https://www.oceanscan-mst.com/">
    <img height="60" alt="REMARO Logo" src="https://isola-project.eu/wp-content/uploads/2020/07/OceanScan.png">
</a>

This work is part of the Reliable AI for Marine Robotics (REMARO) Project. For more info, please visit: <a href="https://remaro.eu/">https://remaro.eu/

<br>

<a href="https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-2020_en">
    <img align="left" height="60" alt="EU Flag" src="https://remaro.eu/wp-content/uploads/2020/09/flag_yellow_low.jpg">
</a>

This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 956200.


