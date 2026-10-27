# flood-mapping-satellite
Flood Mapping Using Satellite Images

1. Problem Statement

Floods are among the most damaging natural disasters. Optical satellite images can be obstructed by clouds during storms, whereas Synthetic Aperture Radar (SAR) can operate during both day and night and is less affected by cloud cover.

Open water often appears dark in SAR imagery because it produces low radar backscatter. The objective of this project is to automatically generate pixel-level water and flood maps from Sentinel-1 SAR satellite images.

2. Methods

The project compares classical SAR thresholding techniques with a deep learning segmentation model.

* Fixed Thresholding: Identifies potential water pixels using a fixed VV backscatter threshold of -15 dB.
* Otsu Thresholding: Automatically determines a threshold from the VV backscatter distribution.
* U-Net: Uses a convolutional neural network for pixel-level water segmentation from Sentinel-1 VV and VH bands.

Evaluation Metrics

The methods are evaluated using:

* Intersection over Union (IoU)
* F1-score
* Precision
* Recall

Metrics are calculated over valid labelled pixels, excluding pixels marked as no-data.

3. Dataset Source

This project uses the Sen1Floods11 dataset, containing Sentinel-1 SAR and Sentinel-2 optical imagery, with 512 × 512 image chips covering 11 global flood events. The dataset includes 446 hand-labelled chips.

* Dataset repository: https://github.com/cloudtostreet/Sen1Floods11
* Google Cloud Storage: gs://sen1floods11
* Research paper: Bonafilia et al., Sen1Floods11, CVPR Workshops, 2020.

Download the dataset separately and configure the data paths according to the downloaded directory structure.

4. How to Run

Step 1: Clone the repository

git clone https://github.com/JAISR-29/flood-mapping-satellite.git
cd flood-mapping-satellite

Step 2: Install dependencies

pip install -r requirements.txt

Step 3: Download the dataset

Follow the dataset instructions in DATASET.md and place the files in the expected directory structure.

Step 4: Run the notebook

jupyter notebook notebooks/flood_mapping.ipynb

Run the notebook cells in order, from data loading through evaluation and visualization.

Requirements

The project uses the following Python libraries:

* NumPy
* Pandas
* Matplotlib
* Rasterio
* Scikit-image
* PyTorch
* Jupyter Notebook

5. Results and Visualizations

Running the notebook generates visualizations and evaluation results in the results/ directory.

Expected output files include:

* 01_samples.png — Sample SAR images and ground-truth masks.
* 02_training_curves.png — Training and validation loss and IoU curves.
* 03_method_comparison.png — Comparison of baseline methods and U-Net.
* 04_predictions.png — Ground-truth masks, predicted water maps, and error visualizations.
* test_metrics.csv — Test-set evaluation metrics.

Actual IoU, F1-score, precision, and recall values will be added after successful execution.

6. Repository Structure

flood-mapping-satellite/
├── notebooks/
│   └── flood_mapping.ipynb
├── results/
├── data/
├── DATASET.md
├── requirements.txt
└── README.md

The dataset is not committed to the repository and must be downloaded separately.

7. Limitations and Future Work

Limitations

* Radar shadow in hilly terrain may cause incorrect water predictions.
* Smooth surfaces such as roads and sand can resemble water in SAR images.
* Urban structures can produce complex radar backscatter patterns.
* Prediction accuracy depends on the quality and coverage of labelled training data.

Future Work

* Combine Sentinel-1 SAR with Sentinel-2 optical indices such as NDWI and MNDWI.
* Experiment with deeper backbones, such as ResNet, and architectures such as U-Net++.
* Evaluate Dice loss and focal loss for improved segmentation.
* Explore pretraining using weakly labelled data.
* Evaluate the model on additional flood events and geographic regions.

8. Conclusion

This project investigates automated flood and water mapping using Sentinel-1 SAR imagery. Classical thresholding methods provide baseline results, while a U-Net model learns spatial features for pixel-level segmentation. Performance is assessed using IoU, F1-score, precision, and recall to compare the methods.
