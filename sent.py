import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter 
import re
import string
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# =================================================================
# STAGE 1: THE PREPROCESSOR CLASS
# =================================================================
# We define this at the top so the rest of the code knows it exists (prevents NameError).

class TextPreprocessor:
    def __init__(self):
        # Compile regex once for efficiency - This is a Google-level optimization.
        # It saves time by not recalculating the pattern for every single review.
        self.html_pattern = re.compile('<.*?>')
        
    def clean_text(self, text):
        """
        Taking messy reviews and turning them into clean words.
        1. Check if it is text.
        2. Strip HTML tags (formatting is not useful language).
        3. Convert to lowercase (Love and LOVE should count as the same).
        4. Remove punctuation (Amazing!!! becomes amazing).
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Remove HTML tags
        text = re.sub(self.html_pattern, '', text)
        
        # 2. Lowercase
        text = text.lower()
        
        # 3. Remove Special Characters and Punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        
        # 4. Remove extra whitespace
        text = " ".join(text.split())
        return text

    def tokenize(self, text):
        # Splits the clean sentence into individual words (tokens).
        return text.split()

# =================================================================
# STAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
# =================================================================
def perform_eda(df, text_col, label_col):
    """
    Why are we defining this? 
    To understand the dataset before training. 
    If "broken" appears often, we know there are negative reviews.
    """
    print("--- Starting EDA ---")
    
    # 1. Check for Class Imbalance (Are there more Positives than Negatives?)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=label_col, data=df)
    plt.title(f"Class Distribution: {label_col}")
    plt.show()
    
    # 2. Review Length Distribution
    # Helps us see if reviews are very short or very long.
    df['review_len'] = df[text_col].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df['review_len'], bins=50, kde=True, color='blue')
    plt.title("Distribution of Review Lengths (Words)")
    plt.xlabel("Number of Words")
    plt.show()
    
    # 3. Common Keywords (Quick Check)
    # Joining all reviews into one long sentence, then counting words.
    all_words = " ".join(df[text_col].astype(str)).lower().split()
    
    # "From the collections toolbox, bring me the Counter tool."
    common_words = Counter(all_words).most_common(20)
    
    print("\nTop 20 Keywords found in dataset:")
    for word, count in common_words:
        print(f"{word}: {count}")

# =================================================================
# STAGE 3: DATA SPLITTING & MODELING
# =================================================================

def train_baseline(X_train, X_test, y_train, y_test):
    """
    Builds a baseline model using TF-IDF and Logistic Regression.
    Establishing a baseline allows us to compare performance later.
    """
    print("\n--- Training Logistic Regression Baseline ---")
    
    # The Pipeline combines TF-IDF (Text to Numbers) and Logistic Regression (The Brain).
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')),
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    # Step 6: Model Training - Studies reviews to learn weights.
    pipeline.fit(X_train, y_train)
    
    # Step 7: Prediction - Predict sentiments for reviews never seen before.
    y_pred = pipeline.predict(X_test)
    
    # Step 8: Model Evaluation
    print("\nBaseline Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return pipeline

def show_top_words(model, vectorizer, n=10):
    """
    Explains why the model predicts positive or negative.
    Extracts 'coefficients' (weights) learned by the model.
    """
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    word_importance = pd.DataFrame({'word': feature_names, 'weight': coefficients})
    
    print("\nTop 10 POSITIVE Words (Highest Weights):")
    print(word_importance.sort_values(by='weight', ascending=False).head(n))
    
    print("\nTop 10 NEGATIVE Words (Lowest Weights):")
    print(word_importance.sort_values(by='weight', ascending=True).head(n))

# =================================================================
# STAGE 4: MAIN EXECUTION (THE GLUE)
# =================================================================
if __name__ == "__main__":
    # Use 'r' before the path to handle backslashes in Windows
    file_path = r"C:\Users\laksh_6eiothu\Downloads\7817_1.csv.zip"

    try:
        # 1. Load Dataset
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        
        # Define the correct column names from your dataset
        target_text_col = 'reviews.text' 
        target_rating_col = 'reviews.rating' 
        
        # 2. Label Engineering
        # Drop rows with missing values
        df = df.dropna(subset=[target_text_col, target_rating_col])
        
        # Remove neutral (3-star) reviews for binary classification
        df = df[df[target_rating_col] != 3].copy()
        
        # Convert ratings to labels: 4-5 = Positive (1), 1-2 = Negative (0)
        df['label'] = df[target_rating_col].apply(lambda x: 1 if x >= 4 else 0)
        
        # 3. Run EDA
        perform_eda(df, text_col=target_text_col, label_col='label')
        
        # 4. Clean Reviews
        print("\nCleaning text data (this may take a moment)...")
        tp = TextPreprocessor() # This is now defined and won't cause NameError
        df['cleaned_text'] = df[target_text_col].apply(tp.clean_text)
        
        # 5. Split Dataset (80% Train, 20% Test)
        # stratify=df['label'] preserves the class ratio (Google best practice).
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], 
            df['label'], 
            test_size=0.2, 
            random_state=42, 
            stratify=df['label']
        )
        
        # 6. Train and Evaluate
        model_pipeline = train_baseline(X_train, X_test, y_train, y_test)
        
        # 7. Show Important Words
        show_top_words(model_pipeline.named_steps['lr'], model_pipeline.named_steps['tfidf'])
        
        # 8. Live Prediction Test
        sample_reviews = [
            "This product is absolutely amazing, I love it!",
            "Terrible quality. It broke the first day. I want a refund."
        ]
        
        print("\n--- Live Test Predictions ---")
        for rev in sample_reviews:
            clean_rev = tp.clean_text(rev)
            pred = model_pipeline.predict([clean_rev])[0]
            sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
            print(f"Review: '{rev}' -> Predicted: {sentiment}")

        # 9. Save the Model
        joblib.dump(model_pipeline, "sentiment_model.pkl")
        print("\nModel saved as 'sentiment_model.pkl'. Project Complete!")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the location.")
    except Exception as e:
        print(f"An error occurred: {e}")