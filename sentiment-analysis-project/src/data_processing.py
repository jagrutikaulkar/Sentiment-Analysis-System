import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Function to lemmatize text
def lemmatize_text(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Function to preprocess the dataset
def preprocess_data(file_path):
    df = pd.read_excel(file_path)  # Load dataset
    df = df.dropna()  # Remove missing values
    
    if 'Review' not in df.columns:
        raise ValueError("Dataset must contain a 'Review' column.")

    df['Cleaned_Review'] = df['Review'].apply(clean_text)
    df['Cleaned_Review'] = df['Cleaned_Review'].apply(remove_stopwords)
    df['Cleaned_Review'] = df['Cleaned_Review'].apply(lemmatize_text)

    # Save cleaned data for training
    df.to_csv('C:/Users/DELL/Desktop/sentiment-analysis-project/data/processed/cleaned_reviews.csv', index=False)
    print("Preprocessed data saved to 'data/processed/cleaned_reviews.csv'")

    return df

# Run this file directly to test preprocessing
if __name__ == "__main__":
    file_path = "C:/Users/DELL/Desktop/sentiment-analysis-project/data/raw/DataSet_PBL_Final_2024-25.xlsx"  # Adjust path as needed
    preprocess_data(file_path)
