# Install dependencies (uncomment if not installed)
# !pip install gradio scikit-learn pandas openpyxl

# Imports
import gradio as gr
import pandas as pd
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Global variables
lr_model = None
mnb_model = None
vectorizer = None
lr_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
mnb_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
training_complete = False
df_sample = None

# Load Positive & Negative Word Corpus
try:
    with open('Positive_Urdu_Words_Corpus.txt', 'r', encoding='utf-8') as f:
        positive_words = set(f.read().splitlines())
    with open('Negative_Urdu_Words_Corpus.txt', 'r', encoding='utf-8') as f:
        negative_words = set(f.read().splitlines())
except FileNotFoundError as e:
    positive_words = set()
    negative_words = set()
    print(f"Warning: Corpus file not found - {e}")

# Training function with progress
def train_model():
    global lr_model, mnb_model, vectorizer, lr_metrics, mnb_metrics, training_complete, df_sample

    try:
        # Check dataset file
        if not os.path.exists('All_Drama_Sentiment_Analysis_UPDATED.xlsx'):
            yield "❌ Error: Dataset file 'All_Drama_Sentiment_Analysis_UPDATED.xlsx' not found!", None, gr.update(interactive=False)
            return

        yield "📖 Loading dataset...", None, gr.update(interactive=False)
        time.sleep(1)
        df = pd.read_excel('All_Drama_Sentiment_Analysis_UPDATED.xlsx')
        df_sample = df[['URDU SENTENCE', 'ACTUAL SENTIMENT']].sample(5, random_state=42).to_html(classes="table-auto w-full border-collapse border border-gray-300", index=False)

        yield "🔍 Extracting features...", None, gr.update(interactive=False)
        time.sleep(1)
        X = df['URDU SENTENCE']
        y = df['ACTUAL SENTIMENT']
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
        X_tfidf = vectorizer.fit_transform(X)

        yield "📈 Splitting data...", None, gr.update(interactive=False)
        time.sleep(1)
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

        # Train Logistic Regression
        yield "🤖 Training Logistic Regression...", None, gr.update(interactive=False)
        time.sleep(1)
        lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
        lr_model.fit(X_train, y_train)

        yield "🤖 Training Multinomial Naive Bayes...", None, gr.update(interactive=False)
        time.sleep(1)
        mnb_model = MultinomialNB()
        mnb_model.fit(X_train, y_train)

        yield "📊 Evaluating models...", None, gr.update(interactive=False)
        time.sleep(1)
        # Evaluate Logistic Regression
        lr_pred = lr_model.predict(X_test)
        lr_metrics['accuracy'] = accuracy_score(y_test, lr_pred)
        lr_metrics['precision'] = precision_score(y_test, lr_pred, average='macro', zero_division=0)
        lr_metrics['recall'] = recall_score(y_test, lr_pred, average='macro', zero_division=0)
        lr_metrics['f1'] = f1_score(y_test, lr_pred, average='macro', zero_division=0)

        # Evaluate Multinomial Naive Bayes
        mnb_pred = mnb_model.predict(X_test)
        mnb_metrics['accuracy'] = accuracy_score(y_test, mnb_pred)
        mnb_metrics['precision'] = precision_score(y_test, mnb_pred, average='macro', zero_division=0)
        mnb_metrics['recall'] = recall_score(y_test, mnb_pred, average='macro', zero_division=0)
        mnb_metrics['f1'] = f1_score(y_test, mnb_pred, average='macro', zero_division=0)

        training_complete = True
        yield f"✅ Training complete! Sample sentences from dataset:", df_sample, gr.update(interactive=True)

    except Exception as e:
        yield f"❌ Error during training: {str(e)}", None, gr.update(interactive=False)

# Sentiment evaluation function
def evaluate_sentiment(urdu_input):
    if not training_complete:
        return "❌ Please complete training first!"

    input_tfidf = vectorizer.transform([urdu_input])
    lr_pred = lr_model.predict(input_tfidf)[0]
    lr_proba = lr_model.predict_proba(input_tfidf)[0]
    mnb_pred = mnb_model.predict(input_tfidf)[0]
    mnb_proba = mnb_model.predict_proba(input_tfidf)[0]

    # Corpus Word Matching
    words = urdu_input.split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)

    if pos_count > neg_count:
        predicted_word = 'Positive'
    elif neg_count > pos_count:
        predicted_word = 'Negative'
    else:
        predicted_word = 'Neutral'

    # Boost Corpus Sentiment if Model unsure
    final_pred = lr_pred
    if lr_pred == 'Neutral' and (pos_count != neg_count):
        final_pred = predicted_word

    # Format Result
    result = f"""
## 📝 User Input:
{urdu_input}

## 🔮 Final Sentiment Prediction:
`{final_pred}`

`{predicted_word}` (Positive Words: {pos_count}, Negative Words: {neg_count})

## ✅ Model Performance Metrics (Test Set):

### Multinomial Naive Bayes:
- Accuracy: **{mnb_metrics['accuracy']:.2f}**
- Precision: **{mnb_metrics['precision']:.2f}**
- Recall: **{mnb_metrics['recall']:.2f}**
- F1 Score: **{mnb_metrics['f1']:.2f}**
"""
    return result

# Testing screen function
def show_testing_screen():
    if not training_complete:
        return "❌ Please complete training first!", gr.update(visible=False)

    metrics = f"""
## 📊 Model Performance Metrics (Test Set):

- Accuracy: **{mnb_metrics['accuracy']:.2f}**
- Precision: **{mnb_metrics['precision']:.2f}**
- Recall: **{mnb_metrics['recall']:.2f}**
- F1 Score: **{mnb_metrics['f1']:.2f}**
"""
    return metrics, gr.update(visible=True)

# Custom CSS for vibrant GUI
custom_css = """
body { background: linear-gradient(to right, #4facfe, #00f2fe); }
.gradio-container { background: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
h1 { color: #1e40af; font-size: 2.5em; text-align: center; }
button { transition: all 0.3s ease; }
button:hover { transform: scale(1.05); }
#train-btn { background: #3b82f6; }
#test-btn { background: #10b981; }
#analyze-btn { background: #8b5cf6; }
textarea { border: 2px solid #60a5fa; border-radius: 10px; padding: 10px; }
.table-auto th, .table-auto td { border: 1px solid #d1d5db; padding: 8px; text-align: left; }
.table-auto th { background: #dbeafe; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
#loading-spinner { animation: spin 1s linear infinite; }
"""

# Gradio GUI
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("<h1>🌟 Urdu Sentiment Analysis</h1>")
    gr.Markdown("Train Logistic Regression and Naive Bayes models to analyze sentiments in Urdu sentences with a vibrant interface!")

    with gr.Row():
        train_btn = gr.Button("TRAINING", elem_id="train-btn")
        test_btn = gr.Button("TESTING", elem_id="test-btn", interactive=False)

    with gr.Column(visible=True) as training_output:
        training_status = gr.Markdown()
        sample_sentences = gr.HTML()
        gr.Markdown("<div id='loading-spinner' class='h-8 w-8 border-4 border-t-blue-500 rounded-full hidden'></div>")

    with gr.Column(visible=False) as testing_output:
        metrics_display = gr.Markdown()
        urdu_input = gr.Textbox(lines=3, placeholder="✨ Enter Urdu Sentence here...", label="Urdu Input")
        analyze_btn = gr.Button("🔍 Analyze Sentiment", elem_id="analyze-btn")
        sentiment_result = gr.Markdown()

    # Event handlers
    def handle_training():
        for status, sample_html, test_btn_state in train_model():
            yield status, sample_html, test_btn_state

    train_btn.click(
        fn=handle_training,
        outputs=[training_status, sample_sentences, test_btn]
    )

    test_btn.click(
        fn=show_testing_screen,
        outputs=[metrics_display, testing_output]
    )

    analyze_btn.click(
        fn=evaluate_sentiment,
        inputs=urdu_input,
        outputs=sentiment_result
    )

# Launch app
demo.launch()
