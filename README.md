# 📰 Fake News Detection using Machine Learning

This project aims to detect fake news using machine learning and natural language processing (NLP) techniques. It is built using Python and scikit-learn and is intended as a beginner-friendly end-to-end classification project.

---

## ✅ Step 1: Dataset Source

We use the **Fake and Real News Dataset** available on Kaggle:  
🔗 [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset]

This dataset contains two CSV files:
- `Fake.csv` — News articles labeled as fake
- `True.csv` — News articles labeled as real

### 🧾 Download Instructions
1. Visit the Kaggle dataset link above.
2. Click **Download** and extract the ZIP file.
3. Move `Fake.csv` and `True.csv` to your project's `/data` directory.

---

## 🧠 Step 2: Load and Combine the Data

We use `pandas` to load both datasets and combine them into a single DataFrame.

### 🔍 Steps:
- Load `Fake.csv` and `True.csv` using `pd.read_csv()`
- Assign a label column:
  - `0` for Fake news
  - `1` for Real news
- Combine both datasets using `pd.concat()`
- Shuffle the combined dataset for randomness

---

---

## 🧹 Step 3: Preprocess the Text

We clean and prepare the news article text to make it suitable for machine learning models.

### 🔍 Steps:
- Convert text to **lowercase**
- Remove **punctuation and special characters**
- Remove **stopwords** using NLTK
- Apply **lemmatization** to reduce words to their base form
- Eliminate any **extra whitespace**

The cleaned output is stored in a new column called `clean_text`.

---

## 📊 Step 4: Vectorize the Text using TF-IDF

We convert the cleaned news article text into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

### 🔍 Steps:
- Loaded the cleaned dataset from `cleaned_news.csv`
- Removed rows with missing values in the `clean_text` column
- Used `TfidfVectorizer` from `sklearn` to convert text into a sparse matrix of TF-IDF features
- Split the data into **training and testing sets** using `train_test_split` (80/20 split)

This step prepares the feature matrix (`X`) and target labels (`y`) for training machine learning models in the next step.

---

## 🧠 Step 5: Train and Evaluate the Model

1. We trained a **Logistic Regression** model to classify news articles as either **real (1)** or **fake (0)** based on their TF-IDF vectorized text features.

### 🔍 Steps:
- Trained the model using `LogisticRegression` from scikit-learn
- Evaluated using:
  - ✅ Accuracy
  - 🎯 Precision
  - 🔁 Recall
  - 🏆 F1 Score

### 📈 Results:
- Accuracy: **~98.45%**
- Precision, Recall, and F1-Score: **~98%** for both classes
- Confusion Matrix shows the model performs well with minimal misclassification

---

### 📊 Visualizations Included:
- **Confusion Matrix** heatmap with class-wise annotations (True Positives, False Negatives, etc.)
- **Bar chart** comparing accuracy, precision, recall, and F1-score
- **Correlation matrix** showing weak correlation between article length and label
- **Top keywords** for fake and real news identified from logistic regression model coefficients

These visualizations help interpret both the overall model performance and individual feature importance in distinguishing fake and real news articles.

---

2. After vectorizing the cleaned news text using TF-IDF, we trained a **Multinomial Naive Bayes (MNB)** classifier to detect whether an article is fake or real.

### 🔍 Why Naive Bayes?
- Designed for text classification tasks
- Fast and efficient even on large datasets
- Performs well with TF-IDF or count-based feature vectors

### 🧪 Steps Performed:
- Trained `MultinomialNB` on the training set
- Predicted labels for the test set
- Evaluated the model using:
  - ✅ Accuracy
  - 🎯 Precision
  - 🔁 Recall
  - 🏆 F1 Score

### 📈 Results:
- Accuracy: ~96–98%
- Strong performance on both fake and real news classification

### 📊 Visualizations Included:
- Confusion matrix heatmap
- Bar chart comparing evaluation metrics
- Word clouds for fake and real news articles
---

Multinomial Naive Bayes proved to be a fast and reliable baseline model for fake news detection.

---