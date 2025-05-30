# Urdu Sentiment Analysis Using Multinomial Naive Bayes

This project implements a machine learning-based sentiment analysis system for Urdu drama dialogues using TF-IDF vectorization and the **Multinomial Naive Bayes (MNB)** classifier. It achieves an accuracy of **78%** on a large dataset of over **431,000 Urdu sentences**, classifying them as **Positive**, **Negative**, or **Neutral**.

---

## 📌 Project Overview

- **Original Dataset Size:** 450,116 Urdu sentences collected from subtitles  
- **Cleaned Dataset Size:** 162,634 Urdu sentences after preprocessing  
- **Balanced Sample for Training:** 20,000 positive + 20,000 negative  
- **Model Used:** Multinomial Naive Bayes (primary)  
- **Alternative Model:** Logistic Regression (baseline)  
- **Accuracy:**  
  - MNB: **78%**  
  - Logistic Regression: **71%**  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score  
- **Interface:** Gradio app with live prediction and model statistics  
- **Tooling:** scikit-learn, Pandas, Gradio, Python  

---

## 🧪 Methodology

### 1. **Data Collection**
- Extracted from YouTube subtitles of 3 Urdu dramas:  
  *Bezuban*, *Masters*, and *Kaise Meri Naseeb*

### 2. **Data Preprocessing**
- Cleaning: Noise removal, UTF-8 encoding, punctuation removal  
- Tokenization, normalization, lowercasing  
- Sentiment labeling: Positive (1), Negative (-1), Neutral (0)

### 3. **Feature Extraction**
- TF-IDF vectorization with:
  - N-gram range: (1, 3)
  - Max features: 10,000

### 4. **Model Training**
- **Train/Test Split:** 80% / 20%
- MNB selected for:
  - High-dimensional feature handling
  - Efficiency on text data
  - Simplicity with strong baseline performance

---

## 📊 Model Performance

| Model                 | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Multinomial Naive Bayes | 0.78     | 0.77      | 0.76   | 0.76     |
| Logistic Regression     | 0.71     | 0.70      | 0.74   | 0.73     |

### Confusion Matrix (MNB)

|                 | Pred. Positive | Pred. Negative | Pred. Neutral |
|-----------------|----------------|----------------|----------------|
| **Actual Positive** | 1500           | 300            | 200            |
| **Actual Negative** | 250            | 1550           | 200            |
| **Actual Neutral**  | 300            | 250            | 1450           |

---

## 🖥️ GUI – Gradio App

A custom Gradio interface enables real-time Urdu sentence sentiment analysis, displaying:

- Model prediction
- Lexicon-boosted sentiment
- Model performance metrics
- Sample predictions

Styling is applied via Tailwind CSS for a clean and modern UI.

---

## 🧠 Why Multinomial Naive Bayes?

- Works well with TF-IDF vectors and discrete word frequencies  
- Handles high-dimensional data efficiently  
- Outperformed Logistic Regression in accuracy and F1-score  
- Simple, interpretable, and fast  

---

## 🔍 Errors & Future Improvements

### Observed Issues
- Misclassification of neutral phrases as positive (e.g., polite expressions)
- Difficulty handling sarcasm or context-heavy lines
- Urdu linguistic complexity limits TF-IDF-based modeling

### Future Enhancements
- Integrate **UrduBERT** or LSTM for context-aware analysis
- Apply stemming/lemmatization for better preprocessing
- Expand dataset and label more dialogues
- Explore **multimodal** sentiment using audio/video signals

---

## 📂 File Structure
📦 bilingual-nlp-urdu-sentiment-analysis
┣ 📄 urdu_sentiment_gradio.py
┣ 📄 sentiment_analysis_app.ipynb
┣ 📄 requirements.txt
┣ 📄 All_Drama_Sentiment_Analysis_UPDATED.xlsx
┣ 📄 Positive_Urdu_Words_Corpus.txt
┣ 📄 Negative_Urdu_Words_Corpus.txt


---

## 📸 Screenshots

| App Interface | Training Completed |
|---------------|------------------|
| ![UI](Screenshot%20(367).png) | ![Training](Screenshot%20(368).png) |

| Training Result | Live Output |
|--------------|-------------|
| ![Metrics](Screenshot%20(369).png) | ![Output](Screenshot%20(370).png) |

Testing Result
--------------
![Metrics](Screenshot%20(371).png)

---

## 🔗 Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Bmhzyxw2vOhaIqUdh9wAkfT4rHShCJi0?usp=sharing)

---

## 🔑 Keywords

`Urdu Sentiment Analysis` · `Multinomial Naive Bayes` · `TF-IDF` · `Text Classification` · `NLP in Urdu` · `Urdu Drama Dataset` · `Machine Learning Urdu` · `Gradio App NLP` · `UrduBERT Future Work`

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Contributions

You're welcome to fork the repo, suggest features, or submit pull requests to enhance this project.

---

Created by **Arfa Rumman Khalid**
Department of Computer Science  
University of Engineering and Technology, Lahore
