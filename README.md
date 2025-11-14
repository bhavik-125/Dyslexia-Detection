To make your README look **properly formatted and highlighted**, you need to use **Markdown section headers, code blocks, and bullet formatting**.

Right now your README looks plain because GitHub Markdown requires the correct symbols.

Below is the **fixed, fully formatted, copy-paste-ready version** with proper highlighting, indentation, code blocks, and headers.

---

# Dyslexia Handwriting Reversal Classifier (YES/NO)

This repository contains a binary classifier that detects whether a handwritten character exhibits reversal patterns commonly associated with dyslexic writing. The system uses a convolutional neural network trained on grayscale character images sorted into three categories: Corrected, Normal, and Reversal. These categories are mapped into a binary decision:

* **YES (1): Reversal**
* **NO (0): Corrected or Normal**

The project includes full training, evaluation, and export to TensorFlow Lite for use on edge devices.

---

## Overview

Letter reversals (such as writing “b” instead of “d”) are common indicators studied in early dyslexia screening.
This project focuses on identifying such reversal tendencies in isolated character samples.
It does not provide medical diagnosis.
The model simply classifies input handwriting images into two classes based on reversal characteristics.

---

## Dataset Requirements

The dataset must be organized in the following folder structure:

```
characters_sorted/
 ├── Corrected/
 ├── Normal/
 └── Reversal/
```

During training, the script automatically converts these three folders into a binary format:

* Reversal → **YES (1)**
* Corrected + Normal → **NO (0)**

All images are loaded in grayscale and resized to **64×64** pixels.

---

## Model Description

The classifier is implemented using TensorFlow and Keras.
It uses a compact CNN architecture designed for fast training and inference:

* Rescaling layer
* Three convolutional blocks with 32, 64, and 128 filters
* Max pooling after each block
* Dense layer with 128 units and dropout
* Sigmoid output layer for binary classification

Mixed-precision training is enabled for performance improvement on supported GPUs.

---

## Training

Run the following command:

```bash
python train_yes_no.py
```

The script performs:

1. Loads dataset with 80/20 train-validation split
2. Converts class labels into binary form
3. Trains the CNN for 20 epochs
4. Saves the trained model as `dyslexia_yes_no.h5`
5. Generates a confusion matrix image
6. Exports a TensorFlow Lite model `dyslexia_yes_no.tflite`

---

## Evaluation Results

Confusion matrix:

```
                 Predicted
                NO        YES
True NO       96311       307
True YES        249     10629
```

Performance:

* Accuracy: **~98.5%**
* Precision (YES): **~97.2%**
* Recall (YES): **~97.7%**
* F1 Score (YES): **~97.4%**

---

## TensorFlow Lite Export

A `.tflite` model is produced for edge deployment.

Example inference code:

```python
interpreter = tf.lite.Interpreter("dyslexia_yes_no.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.expand_dims(preprocessed_img, axis=0).astype("float32")

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

prediction = interpreter.get_tensor(output_details[0]['index'])
label = int(prediction > 0.5)
```

---

## Installation

Install required dependencies:

```bash
pip install tensorflow scikit-learn matplotlib seaborn
```

For GPU acceleration:

```bash
pip install tensorflow[and-cuda]
```




