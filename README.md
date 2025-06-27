# Twitter Sentiment Classification with BERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uy_ldbjPvRAsevn8nbfSOkJ-NAIP0beR?usp=sharing)
[![Python](https://img.shields.io/badge/Python_3.9_+-3776AB?logo=python&logoColor=FF6F00)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?logo=tensorflow)](https://tensorflow.org)

<!--
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)
-->


> **Brief Description:** This project demonstrates how to build and train a sentiment analysis model for Twitter data using the BERT (Bidirectional Encoder Representations from Transformers) model.

> The goal is to classify tweets into three sentiment categories: Positive, Negative, and Neutral. The project utilizes the `transformers` library from Hugging Face and is implemented in a Google Colab notebook environment.


## 📊 Dataset

### Dataset Information

*   **Name:** Twitter Entity Sentiment Analysis
*   **Source:** Kagglehub (`jp797498e/twitter-entity-sentiment-analysis`)
*   **Format:** CSV
*   **Size:** The training dataset contains approximately 74,000 samples, and the validation dataset contains approximately 1,000 samples. The relevant columns used are 'Text' for the tweet content and 'Sentiment' for the corresponding sentiment label.
  

## 🧠 Model Architecture

### Model Overview

### BERT

The model used is a pre-trained `bert-base-uncased` model from the Hugging Face library, specifically `BertForSequenceClassification`. This model consists of a BERT base model followed by a linear layer for sequence classification. It is fine-tuned on the Twitter sentiment dataset.


## 📈 Results

After training for 3 epochs, the model achieved the following performance on the validation set:

*   **Validation Loss:** Approximately 0.1190
*   **Validation Accuracy:** Approximately 0.9783

A detailed classification report and confusion matrix are also generated to show precision, recall, and F1-score for each sentiment class.


## ⚙️ Configuration

### How to Run the Code

1.  Open the Google Colab notebook.
2.  Run all the code cells sequentially.
3.  The notebook will handle dataset download, preprocessing, model training, evaluation, and saving.
4.  You can use the last cell to test the trained model with your own text inputs.

### Files

*   `.ipynb`: The main Colab notebook containing all the code.
*   `model_predictions.png`: A figure visualizing the predicted classes using the model.

### Inference

The trained model and tokenizer are saved to the `saved_model` directory. You can load these saved components to perform sentiment prediction on new text data without retraining. A helper function `predict_sentiment(text)` is provided in the notebook for easy inference.

### Saving and Loading the Model

The trained model and tokenizer are saved locally to the `saved_model` directory using `model.save_pretrained()` and `tokenizer.save_pretrained()`. Additionally, the model is saved to Google Drive for persistent storage.

### Confusion Matrix and Misclassified Examples

The notebook generates a confusion matrix to visualize the model's performance across different sentiment classes. It also displays a few examples of misclassified tweets to help understand the types of errors the model makes.

## 🚀 Future Work

*   Experiment with different BERT variants or other transformer models.
*   Perform more extensive hyperparameter tuning.
*   Explore data augmentation techniques to improve robustness.
*   Implement a more sophisticated error analysis to identify patterns in misclassifications.
*   Build a simple web application or API to deploy the model for real-time inference.

---

<div align="center">

**⭐ Star this repo if you found it helpful!**
-->

Made with ❤️ by [Ritanjit](https://github.com/ritanjit)

</div>
