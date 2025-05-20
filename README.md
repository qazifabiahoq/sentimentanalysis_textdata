

# Sentiment Analysis on Text Data: A Detailed Workflow and Findings

## Objective

The goal of this sentiment analysis project was to classify text data into positive, negative, or neutral sentiments. This helps businesses, researchers, or analysts understand public opinion, customer feedback, or user reviews in a structured and quantitative manner. For instance, product managers can identify strengths and weaknesses of products, marketers can measure campaign effectiveness, and social scientists can analyze public mood.

---

## Step 1: Data Collection and Initial Cleaning

The original dataset consisted of 1000 rows of text reviews. Before any analysis, the data was cleaned to ensure consistency and to prepare it for sentiment scoring and modeling:

* **Text normalization:** Removing or correcting misspellings (e.g., “wa” to “was”), standardizing capitalization, and removing unwanted characters or punctuation.
* **Handling missing values:** Verified no missing text entries to avoid errors in analysis.
* **Tokenization and stopword removal 
---

## Step 2: Exploratory Data Analysis (EDA) and Visualization

Before diving into modeling, the text data was explored to understand common themes and word usage:

* **Top words extraction:**
  The 100 most frequent positive and negative words were identified from the corpus. Examples of frequent positive words included:

  * `'ha'`, `'Trust'`, `'great'`
    Frequent negative words included:
  * `'struck'`, `'violence'`, `'timid'`, `'swearing'`

* **Word frequency counts and word clouds:**
  The word clouds highlighted some of the most prominent words overall, including **"Oz"**, **"Right"**, **"Noise"**, and **"Time"** as the largest and most frequent terms. These key words visually represented the dominant vocabulary and helped confirm the richness and variation in the text data.

---

## Step 3: Sentiment Scoring Using VADER

To quantify sentiment intensity at the sentence level, the VADER (Valence Aware Dictionary and sEntiment Reasoner) tool was employed:

* Each sentence in the dataset was passed through VADER's `SentimentIntensityAnalyzer` to obtain scores for:

  * Negative (neg)
  * Neutral (neu)
  * Positive (pos)
  * Compound (an aggregated normalized score between -1 and 1)

* Example sentiment scores for a sentence:
  `neg: 0.077, neu: 0.711, pos: 0.212, compound: 0.995`

* These scores were added as new columns to the dataframe, enabling quantitative comparisons of sentiment across reviews.

---

## Step 4: Sentiment-based Dataframe and Sorting

With the VADER scores integrated:

* The dataframe was sorted by the **compound score** in descending order to identify the most positive sentences.
* Similarly, sorting by negative scores revealed the most negative sentences.

**Examples:**

* Most positive sentence (compound near 1):
  *"MYSTERY MEN has got to be THE stupidest film I've ever seen but what a film! I thought it was fabulous, excellent and impressive. It was funny, well-done, and nice to see ridiculous Super Heroes for a change!"*

* Most negative sentence (compound near -1):
  *"A rating of doe not begin to express how dull ..."*

---

## Step 5: Creating a Sentiment Target Variable

Based on compound scores, a **Target** sentiment column was created with three classes:

* Positive
* Negative
* Neutral

The class distribution was:

| Sentiment | Count |
| --------- | ----- |
| Positive  | 605   |
| Negative  | 353   |
| Neutral   | 42    |

This labeling made the dataset suitable for supervised machine learning.

---

## Step 6: Label Encoding and Data Preparation

The Target labels were encoded numerically for model compatibility:

* `positive` → 2
* `negative` → 0
* `neutral` → 1

Final dataframe had two columns: `Text` (review) and `Target` (encoded sentiment).

---

## Step 7: Train-Test Split

The dataset was split into training and testing sets for model evaluation:

* 80% training (800 samples)
* 20% testing (200 samples)

Feature extraction (e.g., TF-IDF or Count Vectorizer) yielded feature matrices with 17,800 features (words/tokens) per sample.

---

## Step 8: Machine Learning Models and Evaluation

Several classification algorithms were trained and tested on the sentiment classification task:

| Model                  | Description                                                        |
| ---------------------- | ------------------------------------------------------------------ |
| Logistic Regression    | A linear model useful for binary/multiclass classification.        |
| Support Vector Machine | Finds optimal hyperplanes to separate classes.                     |
| K-Nearest Neighbors    | Classifies based on labels of nearest neighbors in feature space.  |
| Decision Tree          | Builds a tree structure to split data based on feature thresholds. |

**Model performance summary:**

| Model                  | Accuracy | Special Notes                                                                 |
| ---------------------- | -------- | ----------------------------------------------------------------------------- |
| Decision Tree          | 0.8889   | Best model; no false negatives; highest correct positive predictions (12/12). |
| Logistic Regression    | 0.8333   | Good baseline; similar accuracy to SVM and KNN.                               |
| Support Vector Machine | 0.8333   | Similar performance to Logistic Regression and KNN.                           |
| K-Nearest Neighbors    | 0.8333   | Same accuracy as above; comparatively less robust.                            |

---

## Step 9: Interpretation of Results and Ranking

* **Decision Tree** was the top performer due to its high accuracy (88.89%) and crucially, zero false negatives. This means all positive sentiments were correctly identified, which is often critical in sentiment analysis to avoid missing positive feedback.

* **Logistic Regression, SVM, and KNN** shared the same accuracy (\~83.33%), making them acceptable but less reliable for this dataset.

* The tree-based approach’s interpretability and performance make it suitable for deployment or further analysis.

---

## Final Summary

1. **Why Sentiment Analysis?**
   Sentiment analysis provides actionable insights from unstructured text, enabling organizations to gauge opinions effectively.

2. **What was done?**
   The text data was cleaned, explored for sentiment-laden vocabulary, scored with VADER, labeled, and prepared for classification.

3. **Key Findings:**

   * Positive words like `'ha'`, `'Trust'`, and `'great'` appeared frequently.
   * Negative words such as `'struck'`, `'violence'`, and `'swearing'` were also prominent.
   * Word clouds revealed that **"Oz"**, **"Right"**, **"Noise"**, and **"Time"** were the largest and most frequent words, highlighting their dominance in the text corpus.
   * Most positive sentences scored near +1 compound; negatives near -1.
   * The sentiment distribution was imbalanced with a majority positive class.

4. **Models and Performance:**
   The Decision Tree classifier emerged as the best-performing model, followed by Logistic Regression, SVM, and KNN. The absence of false negatives made Decision Tree preferable for sensitive sentiment classification tasks.


## Future Recommendations
To improve the sentiment analysis pipeline and ensure more robust performance, the following steps will be considered in future iterations:

Address Class Imbalance
Resampling techniques such as SMOTE (Synthetic Minority Over-sampling Technique) or class weighting will be explored to reduce bias toward the dominant sentiment class and improve classification performance for minority classes like neutral.

Integrate Advanced NLP Models
Transformer-based models such as BERT or RoBERTa will be tested to capture contextual relationships and semantic nuances that traditional models may miss.

Enhance Text Preprocessing
Future preprocessing will include lemmatization, named entity recognition (NER), and part-of-speech tagging to improve feature quality and retain important linguistic structures.

Experiment with Ensemble Methods
Ensemble models like Random Forests or Voting Classifiers will be implemented to combine strengths of individual classifiers and increase predictive accuracy.

Incorporate Temporal Analysis
If timestamped text data becomes available, sentiment trends over time will be analyzed to identify shifts in public opinion or customer behavior.

Domain-Specific Optimization
The model will be adapted for specific industries by incorporating custom lexicons or fine-tuning on domain-relevant vocabulary to improve relevance and accuracy.

Improve Model Interpretability
Explainability tools such as SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-Agnostic Explanations) will be applied to make individual predictions more transparent and understandable.

---
