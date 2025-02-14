import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Configure the page
st.set_page_config(page_title="Sentiment Analysis", page_icon="🔍", layout="wide")

# Download NLTK Lexicon for VADER Sentiment Analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Define categories for classification
categories = ['Placements', 'Canteen', 'Library', 'Campus']

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Analyze Reviews", "View Insights", "Suggestions", "About Us"],
        icons=["house", "search", "bar-chart", "lightbulb", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Home Page
if selected == "Home":
    st.title("🔍 Sentiment Analysis of College Reviews")
    st.subheader("Upload reviews about your college to analyze sentiments and gain insights.")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file containing reviews:", type="csv")
    if uploaded_file is not None:
        try:
            # Load the file into a DataFrame and save it to session state
            st.session_state['df'] = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error("An error occurred while reading the file. Please ensure it is a valid CSV.")

# Analyze Reviews Page
elif selected == "Analyze Reviews":
    st.header("Analyze Reviews")

    # Check if data exists in session state
    if 'df' in st.session_state:
        df = st.session_state['df']

        # Check if the 'Review' column exists
        if 'Review' in df.columns:
            # Perform sentiment analysis
            def analyze_sentiments(text):
                sentiment = sia.polarity_scores(text)
                if sentiment['compound'] > 0.05:
                    return 'Positive'
                elif sentiment['compound'] < -0.05:
                    return 'Negative'
                else:
                    return 'Neutral'

            # Apply sentiment analysis
            df['Sentiment'] = df['Review'].apply(analyze_sentiments)

            # Overall sentiment distribution
            overall_sentiments = df['Sentiment'].value_counts()

            # Categorize reviews based on keywords
            category_sentiments = {category: {'Positive': 0, 'Negative': 0, 'Neutral': 0} for category in categories}

            for _, row in df.iterrows():
                for category in categories:
                    if category.lower() in row['Review'].lower():
                        category_sentiments[category][row['Sentiment']] += 1

            # Create a DataFrame for category sentiments and save it in session state
            sentiment_data = pd.DataFrame(category_sentiments).T.reset_index()
            sentiment_data.columns = ['Category', 'Positive', 'Negative', 'Neutral']
            st.session_state['sentiment_data'] = sentiment_data

            # Display Graph 1: Overall Sentiments
            st.write("### Overall Sentiment Distribution")
            fig1, ax1 = plt.subplots()
            overall_sentiments.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax1)
            ax1.set_title("Overall Sentiments")
            ax1.set_ylabel("Count")
            ax1.set_xticks(range(len(overall_sentiments)))
            ax1.set_xticklabels(overall_sentiments.index, rotation=0)

            # Add count labels (units) to the right of the bars
            for i, count in enumerate(overall_sentiments):
                ax1.text(i, count + 1, str(count), ha='center', va='bottom')

            st.pyplot(fig1)

            # Display Graph 2: Sentiments by Categories
            st.write("### Sentiment Distribution by Categories")
            selected_category = st.selectbox("Select a Category to View Sentiments", ['All'] + sentiment_data['Category'].tolist())

            if selected_category == 'All':
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                for _, row in sentiment_data.iterrows():
                    ax2.bar(row['Category'], row['Positive'], color='green', label='Positive')
                    ax2.bar(row['Category'], row['Negative'], color='red', label='Negative', bottom=row['Positive'])
                    ax2.bar(row['Category'], row['Neutral'], color='gray', label='Neutral', bottom=row['Positive'] + row['Negative'])

                ax2.set_title("Sentiments for All Categories")
                ax2.set_ylabel("Count")
                ax2.set_xlabel("Categories")

                st.pyplot(fig2)
            else:
                category_data = sentiment_data[sentiment_data['Category'] == selected_category]
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                ax3.bar(['Positive', 'Negative', 'Neutral'], category_data.iloc[0, 1:], color=['green', 'red', 'gray'])
                ax3.set_title(f"Sentiments for {selected_category}")
                ax3.set_ylabel("Count")

                st.pyplot(fig3)

        else:
            st.error("The uploaded file does not contain a 'Review' column. Please check your file and upload again.")
    else:
        st.warning("Please upload reviews on the Home page to analyze.")

# View Insights Page
elif selected == "View Insights":
    st.header("View Insights")

    if 'df' in st.session_state and 'sentiment_data' in st.session_state:
        sentiment_data = st.session_state['sentiment_data']

        st.write("### Sentiment Insights")
        # Display key metrics in a dashboard format
        total_reviews = len(st.session_state['df'])
        positive_reviews = len(st.session_state['df'][st.session_state['df']['Sentiment'] == 'Positive'])
        negative_reviews = len(st.session_state['df'][st.session_state['df']['Sentiment'] == 'Negative'])
        neutral_reviews = len(st.session_state['df'][st.session_state['df']['Sentiment'] == 'Neutral'])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", total_reviews)
        col2.metric("Positive Reviews", positive_reviews, f"{(positive_reviews / total_reviews) * 100:.2f}%")
        col3.metric("Negative Reviews", negative_reviews, f"{(negative_reviews / total_reviews) * 100:.2f}%")

        # Display a pie chart for overall sentiment distribution (reduced size)
        st.write("### Overall Sentiment Breakdown")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            [positive_reviews, negative_reviews, neutral_reviews],
            labels=['Positive', 'Negative', 'Neutral'],
            autopct='%1.1f%%',
            colors=['green', 'red', 'gray']
        )
        ax.set_title("Sentiment Proportion")
        st.pyplot(fig)

    else:
        st.warning("Please upload and analyze reviews to view insights.")

# Suggestions Page
elif selected == "Suggestions":
    st.header("Suggested Improvements")

    if 'sentiment_data' in st.session_state:
        sentiment_data = st.session_state['sentiment_data']

        st.write("### Areas for Improvement")

        # Display suggestions only for categories with negative sentiment
        for _, row in sentiment_data.iterrows():
            if row['Negative'] > 0:  # Show suggestions only if there are negative sentiments
                st.write(f"#### {row['Category']}")

                with st.container():
                    st.markdown(
                        f"""
                        <div style='background-color: #f8d7da; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
                            <strong>Category:</strong> {row['Category']}<br>
                            <strong>Positive:</strong> {row['Positive']}<br>
                            <strong>Negative:</strong> {row['Negative']}<br>
                            <strong>Suggestions:</strong> 
                            <ul>
                                <li>Focus on addressing negative feedback in this area.</li>
                                <li>Increase efforts to improve facilities or services in {row['Category']}.</li>
                                <li>Consider gathering more feedback from users to pinpoint exact issues.</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.warning("Please upload and analyze reviews to view suggestions.")

# About Us Page
elif selected == "About Us":
    st.header("About Us")
    st.write(
        """
        ### Sentiment Analysis for College Reviews

        This web app is developed to help students and faculty analyze reviews of their college in terms of sentiment.
        
        **Key Features:**
        - Upload reviews in CSV format for analysis.
        - Get insights about the overall sentiment distribution.
        - Identify areas that need improvement based on negative feedback.

        **Technologies Used:**
        - Streamlit for frontend development
        - NLTK for Sentiment Analysis using VADER Lexicon
        - Pandas and Matplotlib for data manipulation and visualization

        **Purpose:**
        - Our goal is to provide actionable insights from college reviews and help improve the student experience.

        **Contact:**
        - If you have any feedback or questions, feel free to reach out!
        """
    )
