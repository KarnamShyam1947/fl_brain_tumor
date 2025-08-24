
# Federated Brain Tumor Detection using Flower Framework

This project implements **Federated Learning** using the **Flower Framework** for detecting brain tumors from MRI images. The goal is to collaboratively train a deep learning model across multiple clients without sharing their local data, preserving data privacy.

---

## 📁 Directory Structure

```

root-folder/
├── brain\_tumor\_fl\_model.h5               # Federated trained model
├── brain\_tumor\_vgg16\_epochs\_10.h5        # Pretrained VGG16 model (centralized baseline)
├── client.py                             # Client 1 code for federated training
├── client1.py                            # Client 2 code for federated training
├── predict.py                            # Script to run predictions using the model
├── server.py                             # Federated learning server code

````

---

## 📦 Requirements

Install the required libraries:

```bash
pip install flwr tensorflow
````

---

## 🚀 How to Run

### Step 1: Start the Flower Server

```bash
python server.py
```

### Step 2: Start the Clients (in separate terminals)

```bash
python client.py
```

```bash
python client1.py
```

### Step 3: Run Predictions (optional)

```bash
python predict.py
```

---

## 📊 Dataset

The MRI brain tumor dataset is sourced from [Kaggle](https://www.kaggle.com/). Download and organize it appropriately before training.

Example Kaggle dataset: [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

---

## 🧠 Model Details

* **VGG16** was used as the base CNN architecture for classification.
* Federated averaging (`FedAvg`) is used to aggregate model updates on the server.
* The model is saved as `brain_tumor_fl_model.h5` after federated training.

---

## 📌 Notes

* This is a simplified simulation of federated learning using local scripts for server and clients.
* Useful for educational and experimental purposes.

