import torch
import pandas as pd
from transformers import BertTokenizer

# Load BERT tokenizer
def load_tokenizer(model_name="bert-base-uncased"):
    """
    Load BERT tokenizer.
    """
    return BertTokenizer.from_pretrained(model_name)

# Tokenize reviews using BERT
def tokenize_texts(texts, tokenizer, max_length=256):
    """
    Tokenize the input texts using BERT tokenizer.
    """
    return tokenizer(
        texts.tolist(), 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    )

if __name__ == "__main__":
    # Load preprocessed data
    processed_data_path = r"C:\Users\DELL\Desktop\sentiment-analysis-project\data\processed\cleaned_reviews.csv"
    
    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print("❌ ERROR: Processed data file not found. Check the path.")
        exit()

    # Ensure 'Cleaned_Review' column exists
    if "Cleaned_Review" not in df.columns:
        print("❌ ERROR: 'Cleaned_Review' column not found in the CSV.")
        print("Columns found:", df.columns)
        exit()

    # Drop NaN values and convert to strings
    df["Cleaned_Review"] = df["Cleaned_Review"].astype(str).fillna("")

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Tokenize reviews
    try:
        tokenized_data = tokenize_texts(df["Cleaned_Review"], tokenizer)
    except ValueError as e:
        print(f"❌ ERROR during tokenization: {e}")
        print("Check if 'Cleaned_Review' column contains non-text values.")
        exit()

    # Save tokenized data
    save_path = r"C:\Users\DELL\Desktop\sentiment-analysis-project\data\processed\tokenized_reviews.pt"
    torch.save(tokenized_data, save_path)
    
    print("✅ Tokenized data saved successfully at:", save_path)
