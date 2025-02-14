import pandas as pd
from textblob import TextBlob

# Load processed data
processed_data_path = r"C:\Users\DELL\Desktop\Sentiment-Analysis-System\data\processed\cleaned_reviews.csv"
labels_save_path = r"C:\Users\DELL\Desktop\Sentiment-Analysis-System\data\processed\sentiment_labels.csv"

try:
    df = pd.read_csv(processed_data_path)
except FileNotFoundError:
    print("❌ ERROR: Processed review file not found.")
    exit()

# Ensure "Cleaned_Review" column exists
if "Cleaned_Review" not in df.columns:
    print("❌ ERROR: 'Cleaned_Review' column missing in processed data.")
    print("Columns found:", df.columns)
    exit()

# Function to generate sentiment labels (Positive: 2, Neutral: 1, Negative: 0)
def get_sentiment(text):
    if pd.isna(text):  # Handle missing values
        return 1  # Default to Neutral if text is missing
    text = str(text)  # Ensure input is a string
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 2  # Positive
    elif polarity == 0:
        return 1  # Neutral
    else:
        return 0  # Negative

# Apply sentiment labeling
df["Sentiment"] = df["Cleaned_Review"].apply(get_sentiment)

# Save labeled data
df[["Cleaned_Review", "Sentiment"]].to_csv(labels_save_path, index=False)
print("✅ Sentiment labels generated and saved successfully at:", labels_save_path)
