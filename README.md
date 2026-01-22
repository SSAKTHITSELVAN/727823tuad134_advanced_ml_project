# Spam Email Detection using Naive Bayes

This project implements a robust machine learning pipeline to classify emails as **Spam** or **Ham** (legitimate). Using Natural Language Processing (NLP) and Naive Bayes classifiers, the system transforms raw text into mathematical features to predict the nature of incoming communication.

## 📌 Project Overview

* **Problem Statement:** Addressing the persistent challenge of unsolicited and malicious emails by building a high-precision text classifier.
* **Core Technology:** Python, Scikit-Learn, NLTK, and Naive Bayes Algorithms.
* **Primary Goal:** To compare and evaluate different Naive Bayes architectures (Multinomial vs. Gaussian) in the context of text classification.

---

## 🛠️ Technical Workflow

### 1. Data Preprocessing

Before training, the text undergoes a rigorous cleaning process:

* **Cleaning:** Removal of URLs, email addresses, special characters, and conversion to lowercase.
* **Tokenization:** Breaking sentences into individual words.
* **Stop Word Removal:** Deleting common words (e.g., "the", "is", "at") that do not carry significant meaning.
* **Stemming:** Reducing words to their root form (e.g., "winning" becomes "win") using the Porter Stemmer.

### 2. Feature Extraction

The cleaned text is converted into numerical data using two primary methods:

* **Bag of Words (BoW):** Counts the frequency of each word in the document.
* **TF-IDF (Term Frequency-Inverse Document Frequency):** Weights words based on their importance across the entire dataset, penalizing words that appear too frequently across all documents.

### 3. Model Implementation

We implemented two variations of the Naive Bayes algorithm:

* **Multinomial Naive Bayes:** Optimized for discrete counts (ideal for text data).
* **Gaussian Naive Bayes:** Used for continuous features, assuming a normal distribution.

---

## 📊 Performance Metrics

The models are evaluated based on four key indicators:
| Metric | Description | Importance in Spam Detection |
| :--- | :--- | :--- |
| **Accuracy** | Overall correct predictions. | General reliability. |
| **Precision** | Ratio of true spam to total predicted spam. | **Critical:** High precision ensures legitimate emails aren't sent to spam. |
| **Recall** | Ratio of true spam detected. | Ensures as many spam emails as possible are caught. |
| **F1-Score** | Harmonic mean of Precision and Recall. | Provides a balanced view of model performance. |

---

## 🚀 Key Deliverables Included

1. **Preprocessing Report:** Detailed steps from raw text to cleaned tokens.
2. **Comparative Analysis:** A side-by-side performance review of Gaussian vs. Multinomial models.
3. **Visual Analytics:** Confusion Matrices and ROC Curves to visualize classification boundaries and error rates.
4. **Live Predictor:** A function to test the model against "unseen" real-world email samples.

---

## 💡 Findings & Recommendations

* **Multinomial NB** consistently outperforms Gaussian NB for text tasks because it handles sparse word counts more naturally.
* **Feature Scaling:** Using TF-IDF generally provides better context than simple word counts.
* **Next Steps:** For future iterations, implementing N-grams (considering sequences of 2 or 3 words) could further improve detection of sophisticated spam phrases.

---

**How to run the project:**

1. Install dependencies: `pip install pandas numpy scikit-learn nltk matplotlib seaborn`
2. Run the Jupyter Notebook or Python script.
3. Check the generated `spam_detection_analysis.png` for a visual summary of the results.

Would you like me to help you create a **Requirements.txt** file to go along with this README?