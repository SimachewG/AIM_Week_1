import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Configure logging
logging.basicConfig(
    filename='news_analysis.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_data(file_path):
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    logging.info("Loading data from file: %s", file_path)
    df = pd.read_csv(file_path)
    logging.info("Data loaded successfully with %d rows and %d columns", df.shape[0], df.shape[1])
    return df

def analyze_headline_length(df):
    """Compute descriptive statistics for the length of headlines.

    Args:
        df (pd.DataFrame): DataFrame containing a 'headline' column.

    Returns:
        pd.Series: Descriptive statistics of headline lengths.
    """
    logging.info("Analyzing headline length statistics.")
    df['headline_length'] = df['headline'].str.len()
    stats = df['headline_length'].describe()
    logging.info("Headline length analysis complete.")
    return stats

def count_articles_per_publisher(df):
    """Count the number of articles per publisher.

    Args:
        df (pd.DataFrame): DataFrame containing a 'publisher' column.

    Returns:
        pd.Series: Count of articles by publisher.
    """
    logging.info("Counting articles per publisher.")
    counts = df['publisher'].value_counts()
    logging.info("Article count per publisher complete.")
    return counts

def analyze_publication_dates(df):
    """Analyze various aspects of publication dates.

    Args:
        df (pd.DataFrame): DataFrame with a 'date' column in datetime format.

    Returns:
        dict: Daily, top days, weekday, and monthly article counts.
    """
    logging.info("Analyzing publication dates.")
    daily_counts = df.groupby(df['date'].dt.date).size()
    top_days = daily_counts.nlargest(5)
    weekday_counts = df['date'].dt.day_name().value_counts()
    monthly_counts = df.groupby(df['date'].dt.to_period('M').dt.to_timestamp()).size()

    logging.info("Publication dates analysis complete.")
    return {
        'daily_counts': daily_counts,
        'top_days': top_days,
        'weekday_counts': weekday_counts,
        'monthly_counts': monthly_counts
    }

def plot_publication_trends(date_analysis):
    """Plot daily, weekly, and monthly publication trends.

    Args:
        date_analysis (dict): Dictionary from analyze_publication_dates().

    Returns:
        matplotlib.figure.Figure: The resulting figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    date_analysis['daily_counts'].plot(ax=axes[0, 0])
    axes[0, 0].set_title('Daily Article Count')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Number of Articles')

    date_analysis['top_days'].plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Top 5 Days with Most Articles')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Number of Articles')

    date_analysis['weekday_counts'].plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Article Distribution by Weekday')
    axes[1, 0].set_xlabel('Weekday')
    axes[1, 0].set_ylabel('Number of Articles')

    monthly_counts = date_analysis['monthly_counts']
    monthly_counts.plot(ax=axes[1, 1])
    axes[1, 1].set_title('Monthly Article Count')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Number of Articles')
    axes[1, 1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig

def perform_sentiment_analysis(df, text_column='headline'):
    """Perform sentiment analysis using VADER.

    Args:
        df (pd.DataFrame): DataFrame with the text column.
        text_column (str): Name of the column containing text data.

    Returns:
        pd.DataFrame: Updated DataFrame with sentiment scores and labels.
    """
    logging.info("Performing sentiment analysis on column: %s", text_column)
    sia = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df[text_column].apply(lambda x: sia.polarity_scores(x))
    df['sentiment'] = df['sentiment_scores'].apply(
        lambda x: 'positive' if x['compound'] > 0 else ('negative' if x['compound'] < 0 else 'neutral')
    )
    logging.info("Sentiment analysis complete.")
    return df

def perform_topic_modeling(df, text_column='headline', num_topics=5, num_words=10):
    """Perform topic modeling using Latent Dirichlet Allocation (LDA).

    Args:
        df (pd.DataFrame): DataFrame with the text column.
        text_column (str): Name of the text column.
        num_topics (int): Number of topics to extract.
        num_words (int): Number of words per topic.

    Returns:
        list: List of topics with top keywords.
    """
    logging.info("Performing topic modeling on column: %s", text_column)
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df[text_column])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [(words[i], topic[i]) for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(top_words)

    logging.info("Topic modeling complete. %d topics identified.", num_topics)
    return topics

def analyze_publication_times(df):
    """Analyze article publication times by hour of the day.

    Args:
        df (pd.DataFrame): DataFrame with a datetime 'date' column.

    Returns:
        str: Peak hour of publication.
    """
    logging.info("Analyzing distribution of publication times.")
    df['hour'] = df['date'].dt.hour
    hourly_distribution = df['hour'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    hourly_distribution.plot(kind='bar')
    plt.title('Distribution of Article Publications by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    peak_hour = hourly_distribution.idxmax()
    logging.info("The peak publication hour is %d:00.", peak_hour)
    return f"The peak publication hour is {peak_hour}:00"

def identify_publication_spikes(df, threshold=2):
    """Identify days with unusually high publication frequency.

    Args:
        df (pd.DataFrame): DataFrame with a datetime 'date' column.
        threshold (float): Number of standard deviations above the mean.

    Returns:
        pd.Series: Days with unusually high publication volume.
    """
    logging.info("Identifying days with unusually high publication frequency.")
    daily_counts = df.groupby(df['date'].dt.date).size()
    mean_publications = daily_counts.mean()
    std_publications = daily_counts.std()
    spikes = daily_counts[daily_counts > mean_publications + threshold * std_publications]
    logging.info("Publication spikes identified: %d days.", len(spikes))
    return spikes

def analyze_publishers(df):
    """Visualize the top 10 publishers by article count.

    Args:
        df (pd.DataFrame): DataFrame with a 'publisher' column.

    Returns:
        pd.Series: Top 10 publishers by number of articles.
    """
    logging.info("Analyzing contribution by different publishers.")
    publisher_counts = df['publisher'].value_counts()
    top_publishers = publisher_counts.head(10)

    plt.figure(figsize=(12, 6))
    top_publishers.plot(kind='bar')
    plt.title('Top 10 Publishers by Number of Articles')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    logging.info("Top 10 publishers analysis complete.")
    return top_publishers

def analyze_publisher_domains(df):
    """Analyze the domain names of publisher identifiers.

    Args:
        df (pd.DataFrame): DataFrame with a 'publisher' column.

    Returns:
        pd.Series: Count of unique domains.
    """
    logging.info("Analyzing unique domains in publisher names.")

    def extract_domain(email):
        try:
            return email.split('@')[1]
        except IndexError:
            return email

    df['domain'] = df['publisher'].apply(extract_domain)
    domain_counts = df['domain'].value_counts()

    plt.figure(figsize=(12, 6))
    domain_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Publisher Domains')
    plt.xlabel('Domain')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    logging.info("Publisher domain analysis complete.")
    return domain_counts

def analyze_news_types_by_publisher(df, top_n=5):
    """Analyze types of news reported by top publishers using word frequency.

    Args:
        df (pd.DataFrame): DataFrame with 'publisher' and 'headline' columns.
        top_n (int): Number of top publishers to analyze.
    """
    logging.info("Analyzing news types reported by top %d publishers.", top_n)
    stop_words = set(stopwords.words('english'))
    top_publishers = df['publisher'].value_counts().head(top_n).index

    for publisher in top_publishers:
        logging.info("Analyzing news type for publisher: %s", publisher)
        publisher_headlines = df[df['publisher'] == publisher]['headline']
        words = []

        for headline in publisher_headlines:
            tokens = word_tokenize(headline.lower())
            words.extend([word for word in tokens if word.isalnum() and word not in stop_words])

        word_freq = Counter(words)
        logging.info("Top 10 most common words for %s: %s", publisher, word_freq.most_common(10))

        print(f"\nTop 10 most common words for {publisher}:")
        for word, count in word_freq.most_common(10):
            print(f"{word}: {count}")
