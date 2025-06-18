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

