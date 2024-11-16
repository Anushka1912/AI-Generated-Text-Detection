# **AI Text Detection Using BERT**

This project uses a fine-tuned BERT model to classify whether a given text is **AI-generated** or **human-written**. The model is trained on a labeled dataset of essays and performs binary classification.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Model Training](#model-training)
7. [Inference](#inference)
8. [Results](#results)



---

## **Project Overview**

This project leverages the **transformers library** by Hugging Face to train a BERT model for detecting AI-generated content. It preprocesses the text, tokenizes it using the BERT tokenizer, and trains a binary classification head on top of BERT.

---

## **Features**
- Data cleaning and preprocessing.
- Class balance check and handling.
- Fine-tuning BERT for sequence classification.
- Model evaluation and accuracy reporting.
- User input for real-time text classification.
- CSV file predictions for test datasets.

---

## **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ai-text-detection.git
   cd ai-text-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. Ensure you have a GPU-enabled environment for faster training:
   - Install PyTorch: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

---

## **Usage**

### **1. Data Preprocessing**
- Place your training and test datasets (`train_essays.csv` and `test_essays.csv`) in the project directory.
- Ensure the datasets include the required columns:
  - `id`: Unique identifier for each text.
  - `text`: The essay content.
  - `generated`: Binary labels (1 = AI-generated, 0 = Human-written) for training data.

### **2. Run Training**
To train the model, execute:
```bash
python main.py
```

### **3. Real-time Text Classification**
After training, you can input custom text for classification:
```python
sample_text = ["This is a sample essay written by AI."]
```

### **4. Generate Predictions**
Save predictions for the test dataset in a CSV file:
```bash
python main.py
# Outputs 'sample_submission.csv' containing the predictions.
```

---

## **Dataset**

### **Format**
The training dataset (`train_essays.csv`) must contain:
- `id`: Unique identifier.
- `text`: Text content.
- `generated`: Binary labels for classification.

The test dataset (`test_essays.csv`) must contain:
- `id`: Unique identifier.
- `text`: Text content.

---

## **Model Training**

The project fine-tunes a pre-trained **BERT base uncased** model from Hugging Face with the following settings:
- Batch size: 16
- Max sequence length: 128
- Learning rate: 2e-5
- Epochs: 10
- Optimizer: AdamW with weight decay

---

## **Inference**

For real-time predictions, modify the `main.py` file to accept user input:
```python
user_input = input("Enter text to classify: ")
tokenized_input = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    output = model(**tokenized_input)
    probs = torch.softmax(output.logits, dim=1)
    print(f"Prediction: {'AI-generated' if probs[0][1] > 0.5 else 'Human-written'}")
```

---

## **Results**

### **Training Accuracy**
The model achieves high accuracy in distinguishing between AI-generated and human-written texts based on the training dataset.

### **Validation Accuracy**
Validation accuracy is reported during training.

### **Example Outputs**
| **Text**                              | **Prediction**    |
|---------------------------------------|-------------------|
| "This is a human-written essay."      | Human-written     |
| "The AI-generated content is clear."  | AI-generated      |

---



