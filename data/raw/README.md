# CIC-IDS2017 Raw Dataset Ingestion Directory

This directory is the configured drop location (`RAW_DATA_DIR`) for the raw CSV files of the **CIC-IDS2017 Dataset**.

### Source Links (Shown in Documentation)
* **Kaggle Mirror (Chethan H N):** [https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
* **Official UNB Portal:** [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
* **Direct Zip Link:** [http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip](http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip)

### Expected CSV Files
When downloaded and extracted from Kaggle or UNB, drop any or all of the following 8 CSV files into this directory:
1. `Monday-WorkingHours.pcap_ISCX.csv`
2. `Tuesday-WorkingHours.pcap_ISCX.csv`
3. `Wednesday-workingHours.pcap_ISCX.csv`
4. `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
5. `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
6. `Friday-WorkingHours-Morning.pcap_ISCX.csv`
7. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
8. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

### Training Command
Once you place the CSV files or download archive here, run:
```bash
python3 -m scripts.train_full_dataset
```
The script will automatically detect the files, run preprocessing, train the Binary, Multiclass, and Anomaly models, and update the model weights in `models/`.

