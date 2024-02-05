# SubPipe-dataset :diving_mask:
A Submarine Pipeline Inspection Dataset for Segmentation and Visual-inertial Localization.
Here you will find all the relevant information regarding the dataset introduced in [our paper:](https://arxiv.org/abs/2401.17907)
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
  - [Acknowledgements](#acknowledgements)


## 1. The dataset

 ### 1.1 Dataset link
 You can download SubPipe from [the following link](https://zenodo.org/records/10053565?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk3YjQ3MDMyLTVkNjQtNGVjZi05YWM0LThmMWViZDdlZjZhYSIsImRhdGEiOnt9LCJyYW5kb20iOiI1OWM2MWFhMGJiM2ExYThiMGZjNzViZjQ3ZTBiZWRmMyJ9.cGHld8zcCv2Un3LWDJo_S8IExiTfaQqyIZusOQ0VGHywkJXM5YiOieUBgyRCgXp7s6kWHKymrOQWnGVu-A2utg)

###  1.2 The dataset structure
The dataset is divided in 5 Chunks. We exemplify SubPipe's structure with Chunk 0 as:

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
│   │   ├── Cam0_images
│   │   │   ├── <timestamp0>.jpg
│   │   │   ├── ...
│   │   │   └── <timestampN>.jpg
│   │   ├── Depth.csv
│   │   ├── AngularVelocity.csv
│   │   ├── Altitude.csv
│   │   └── Acceleration.csv
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
- The results from applying the metrics on state-of-the-art datasets.
- A notebook to plot the metrics.
- The code for deploying the metrics.

The metrics proposed are delentropy and motion diversity. For more info about those metrics, consider reading [our paper](https://arxiv.org/abs/2401.17907).

## Acknowledgements

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


