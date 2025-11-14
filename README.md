Dyslexia Handwriting Reversal Classifier (YES/NO)

This repository contains a binary classifier that detects whether a handwritten character exhibits reversal patterns commonly associated with dyslexic writing. The system uses a convolutional neural network trained on grayscale character images sorted into three categories: Corrected, Normal, and Reversal. These categories are mapped into a binary decision:

YES (1): Reversal
NO (0): Corrected or Normal

The project includes full training, evaluation, and export to TensorFlow Lite for use on edge devices.

Overview

Letter reversals (such as writing “b” instead of “d”) are common indicators studied in early dyslexia screening. This project focuses on identifying such reversal tendencies in isolated character samples. It does not provide medical diagnosis. The model simply classifies input handwriting images into two classes based on reversal characteristics.


Dataset Requirements

The dataset must be organized in the following folder structure:

characters_sorted/
├── Corrected/
├── Normal/
└── Reversal/

During training, the script automatically converts these three folders into a binary format:

Reversal → YES (1)
Corrected + Normal → NO (0)

All images are loaded in grayscale and resized to 64×64 pixels.

Model Description

The classifier is implemented using TensorFlow and Keras. It uses a compact CNN architecture designed for fast training and inference:

Rescaling layer

Three convolutional blocks with 32, 64, and 128 filters

Max pooling after each block

Dense layer with 128 units and dropout

Sigmoid output layer for binary classification

Mixed-precision training is enabled for performance improvement on supported GPUs.

Training

Run the following command:

python train_yes_no.py

The script performs:

Loads dataset with 80/20 train-validation split

Converts class labels into binary form

Trains the CNN for 20 epochs

Saves the trained model as dyslexia_yes_no.h5

Generates a confusion matrix image

Exports a TensorFlow Lite model (dyslexia_yes_no.tflite)

Evaluation Results

Confusion matrix:

True NO predicted as NO: 96311
True NO predicted as YES: 307
True YES predicted as NO: 249
True YES predicted as YES: 10629

Performance estimates:

Accuracy: ~98.5%
Precision (YES): ~97.2%
Recall (YES): ~97.7%
F1 Score (YES): ~97.4%

The model shows strong ability to separate reversal from non-reversal handwriting with low error rates.

TensorFlow Lite Export

The training script generates dyslexia_yes_no.tflite, which can be deployed on Raspberry Pi, mobile devices, and embedded systems.

Example inference:

interpreter = tf.lite.Interpreter("dyslexia_yes_no.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.expand_dims(preprocessed_img, axis=0).astype("float32")

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

prediction = interpreter.get_tensor(output_details[0]['index'])
label = int(prediction > 0.5)

Installation

pip install tensorflow scikit-learn matplotlib seaborn

For GPU support:

pip install tensorflow[and-cuda]
