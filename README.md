# Spam SMS Detection using Machine Learning

This project demonstrates the classification of SMS messages as **Spam** or **Ham** using Machine Learning techniques. The dataset used is the **SMS Spam Collection Dataset** from Kaggle, and the solution is implemented in Python with popular libraries such as `pandas`, `scikit-learn`, and `nltk`.

## Project Overview

The goal of this project is to:
1. **Clean and preprocess** SMS messages
2. **Vectorize** the text using techniques like Count Vectorization
3. **Train and evaluate** a machine learning model (Naive Bayes)
4. **Predict** whether a new message is Spam or Ham
5. **Visualize** the model performance

---

## Steps Performed

### 1. **Data Loading and Exploration**
- Loaded the dataset from Kaggle
- Explored the distribution of spam and ham messages

### 2. **Text Preprocessing**
- Converted text to lowercase
- Removed **special characters**, **numbers**, and **extra spaces**
- Used **nltk**'s stopword removal and **stemming** to reduce words to their base form

### 3. **Feature Engineering**
- Used **CountVectorizer** to convert text messages into a numerical feature matrix (Bag-of-Words model)
  
### 4. **Model Training**
- Trained a **Multinomial Naive Bayes** model on the vectorized data
- Split the data into **80% training** and **20% testing** sets

### 5. **Model Evaluation**
- Evaluated the model using key metrics:
  - **Accuracy**: ~97.7%
  - **Precision**, **Recall**, and **F1-Score** for both Spam and Ham messages
- Visualized the **Confusion Matrix**, **ROC Curve**, and **Precision-Recall Curve**

### 6. **Prediction Function**
- Created a function to predict whether a new SMS message is **Spam** or **Ham**
- Tested with sample messages to validate model performance

### 7. **Next Steps**
- Considered using **TF-IDF Vectorization** instead of **CountVectorizer** for potentially improved model performance.
- Optionally, additional models (e.g., Logistic Regression, SVM) and further evaluation can be explored.
  
---

## Libraries Used
- `pandas`
- `nltk`
- `sklearn`
- `matplotlib`
- `seaborn`

## Instructions to Run

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/spam-sms-detection.git
cd spam-sms-detection



# Sample code to test the model on a new message

text = "Congratulations! You've won a free ticket. Call now!"
prediction = predict_message(text)
print(f"Prediction: {prediction}")
