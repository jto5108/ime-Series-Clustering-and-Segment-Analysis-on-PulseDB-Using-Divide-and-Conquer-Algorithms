Pulse Clustering Project
Project Description

This project performs analysis on arterial blood pressure (ABP) signals from the VitalDB dataset. It clusters segments of ABP signals using a divide-and-conquer approach, identifies the closest pairs within each cluster using DTW distance, and detects maximum subarray intervals using Kadane’s algorithm. Visualizations are generated for clusters, closest pairs, and significant signal intervals.

How to Use

Download the Dataset

Go to the Kaggle dataset page and download VitalDB_AAMI_Test_Subset.mat.

Add it to the Project

Place the downloaded dataset in your project folder.

Update File Path

Open pulse_clustering_db.py and update the dataset_path variable:

dataset_path = 'path_to_your_dataset/VitalDB_AAMI_Test_Subset.mat'


Install Requirements

Make sure you have Python 3.10 or later installed.

Install dependencies using pip:

pip install -r requirements.txt


Run the Program

Execute the script:

python pulse_clustering_db.py


The script will output clusters, closest pairs, Kadane analysis results, and visualizations in PNG format.
