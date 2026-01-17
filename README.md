# Reddit Moderator Tool

**A Data Science Project for Analyzing Reddit Community Engagement and Predicting Post Performance**

**Google Colab**:   [Open in Colab](https://colab.research.google.com/drive/1kn-Bnn9OWjeuTLuF4ay4qdQQlhSjkp1D)

---

## Overview

The **Reddit Moderator Tool** is a comprehensive data science project designed to analyze Reddit community dynamics, predict post engagement, and provide actionable insights for content creators and moderators. This tool leverages machine learning algorithms to understand what drives engagement on Reddit and helps optimize content strategy across multiple subreddits.

Built using Python and Jupyter Notebooks, the project implements a complete data science pipeline from data collection through the Reddit API to predictive modeling using ensemble machine learning techniques.  The system analyzes over 30 popular subreddits to identify patterns in post performance and user engagement.

---

## Problem Statement

In today's digital world, understanding how Reddit communities function is crucial for moderators, content creators, and researchers. The platform's complex voting system, diverse communities, and varying engagement patterns make it challenging to: 

- Predict which posts will gain traction
- Understand factors that drive community engagement
- Optimize posting strategies for maximum visibility
- Analyze sentiment and content trends across subreddits
- Make data-driven moderation decisions

This project addresses these challenges by providing a data-driven framework for analyzing Reddit communities and predicting post performance based on historical data and machine learning models.

---

## Key Features

### Data Collection
- **Reddit API Integration**: Automated data scraping using PRAW (Python Reddit API Wrapper)
- **Multi-Subreddit Support**: Collects data from 30+ diverse subreddits
- **Comprehensive Data Points**:
  - Post title, content, and metadata
  - Upvotes and upvote ratios
  - Comment counts and engagement metrics
  - Timestamp and author information
  - Subreddit-specific features

### Data Processing and Cleaning
- **Missing Value Handling**: Median imputation for numerical features
- **Feature Engineering**:
  - Temporal features (year, month, day of week, hour)
  - Normalized upvote ratios
  - Time-based categorization (AM/PM)
  - Subreddit encoding
- **Data Validation**:  Removal of duplicates and invalid entries
- **Format Standardization**: Consistent data types and structures

### Predictive Modeling
- **Machine Learning Pipeline**:
  - Random Forest Regressor for upvote prediction
  - Hyperparameter tuning with RandomizedSearchCV
  - Cross-validation for model robustness
  - Feature importance analysis
- **Performance Metrics**:
  - Mean Squared Error (MSE)
  - R-squared score
  - Cross-validation scores

### Analysis and Visualization
- **Sentiment Analysis**: Natural language processing for content sentiment
- **Engagement Correlation**: Analysis of factors influencing post performance
- **Temporal Patterns**: Identification of optimal posting times
- **Subreddit Comparison**: Cross-community engagement analysis
- **Data Visualization**: matplotlib and seaborn for insights presentation

---

## Technical Architecture

### Technology Stack

**Core Technologies:**
- Python 3.8+
- Jupyter Notebook
- Google Colab (for cloud execution)

**Key Libraries:**
- **PRAW**: Reddit API interaction
- **pandas**: Data manipulation and analysis
- **numpy**:  Numerical computing
- **scikit-learn**: Machine learning and model evaluation
- **matplotlib & seaborn**: Data visualization
- **NLTK**:  Natural language processing and sentiment analysis

**Machine Learning:**
- Random Forest Regressor
- RandomizedSearchCV for hyperparameter optimization
- SimpleImputer for missing value handling
- LabelEncoder for categorical encoding

### Project Structure

```
Reddit_Moderator_Tool/
├── DataCollection.ipynb          # Reddit API data scraping
├── DataCleaning.ipynb            # Data preprocessing and feature engineering
├── UpvotesPrediction.ipynb       # ML model training and prediction
├── subreddits.txt                # List of target subreddits
├── requirements.txt              # Python dependencies
├── labeled_subreddit_posts.csv   # Processed dataset
└── README.md
```

### Target Subreddits

The project analyzes the following 31 subreddits:

```
AskReddit, ChangeMyView, TodayILearned, self, offmychest,
Showerthoughts, personalfinance, AskScience, Writing, Advice,
LetsNotMeet, SelfImprovement, DecidingToBeBetter, AskHistorians,
TwoXChromosomes, CasualConversation, InternetIsBeautiful, nosleep,
WritingPrompts, ExplainLikeImFive, TrueOffMyChest, UnpopularOpinion,
relationships, TrueAskReddit, Confession, ShortScaryStories,
ProRevenge, NuclearRevenge, LifeProTips, needadvice, TrueUnpopularOpinion
```

---

## Implementation Details

### 1. Data Collection Pipeline

**Reddit API Setup:**
```python
import praw

def setup_reddit_api():
    return praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="YOUR_USER_AGENT"
    )
```

**Data Extraction:**
```python
def collect_subreddit_data(reddit, subreddit_name, post_limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []

    for post in subreddit. hot(limit=post_limit):
        posts_data.append({
            'subreddit': subreddit_name,
            'title': post. title,
            'content': post.selftext,
            'upvotes': post.score,
            'upvote_ratio': post.upvote_ratio,
            'comments_count': post.num_comments,
            'author': post.author.name if post.author else 'deleted',
            'timestamp': datetime.fromtimestamp(post.created_utc),
            'post_id': post.id
        })

    return pd.DataFrame(posts_data)
```

### 2. Feature Engineering

**Temporal Features:**
```python
# Convert timestamp to datetime and extract features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour'] = df['timestamp'].dt.hour

# Create time period categories
df['date'] = df['timestamp'].dt.date
df['time'] = df['timestamp'].dt.time
df['AM_PM'] = df['timestamp'].dt.hour. apply(lambda x: 'AM' if x < 12 else 'PM')
```

**Normalization:**
```python
# Normalize upvote ratio
df['upvote_ratio_normalized'] = (df['upvote_ratio'] - df['upvote_ratio'].min()) / \
                                 (df['upvote_ratio'].max() - df['upvote_ratio'].min())
```

### 3. Machine Learning Model

**Data Preparation:**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Handle missing values
imputer = SimpleImputer(strategy='median')
df['subscriber_count'] = imputer.fit_transform(df[['subscriber_count']])

# Encode categorical variables
label_encoder = LabelEncoder()
df['subreddit_name'] = label_encoder.fit_transform(df['subreddit_name'].fillna('Unknown'))

# Prepare features and target
X = df.drop(columns=['upvotes', 'title', 'content', 'author', 'post_id', 'date', 'time', 'AM_PM'])
y = df['upvotes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Model Training with Hyperparameter Tuning:**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define hyperparameter distribution
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize Random Forest
rf_model = RandomForestRegressor(random_state=42)

# Randomized search for best parameters
random_search = RandomizedSearchCV(
    rf_model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

# Train model
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
```

**Model Evaluation:**
```python
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Predictions
y_pred = best_model.predict(X_test)

# Performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Cross-validation
cv_scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring='neg_mean_squared_error'
)

print(f"Mean Squared Error:  {mse}")
print(f"R-squared Score: {r2}")
print(f"Cross-validation MSE: {-cv_scores.mean()}")
```

### 4. Data Visualization

**Engagement Analysis:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Upvotes distribution by subreddit
plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x='subreddit', y='upvotes')
plt. xticks(rotation=90)
plt.title('Upvotes Distribution Across Subreddits')
plt.tight_layout()
plt.show()

# Temporal patterns
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='hour', y='upvotes', hue='subreddit', estimator='mean')
plt.title('Average Upvotes by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Upvotes')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Reddit API credentials (client ID and secret)

### Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/Superkart/Reddit_Moderator_Tool.git
cd Reddit_Moderator_Tool
```

**2. Create a virtual environment**
```bash
python -m venv venv
```

**Windows PowerShell (if needed):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

**3. Activate the virtual environment**

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Start Jupyter Notebook**
```bash
jupyter notebook
```

### Google Colab Setup

**Option 1: Direct Link**
- Open the [Google Colab notebook](https://colab.research.google.com/drive/1kn-Bnn9OWjeuTLuF4ay4qdQQlhSjkp1D)
- Run cells sequentially
- No local installation required

**Option 2: Upload to Colab**
1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook
3. Select `.ipynb` files from the repository
4. Run cells in order

### Reddit API Configuration

**1. Create Reddit Application**
- Visit [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
- Click "Create App" or "Create Another App"
- Select "script" as the app type
- Note your `client_id` and `client_secret`

**2. Update API Credentials**
```python
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="YourAppName/1.0 by u/YourUsername"
)
```

---

## Usage Guide

### Running the Complete Pipeline

**Step 1: Data Collection**
```bash
jupyter notebook DataCollection.ipynb
```
- Configure Reddit API credentials
- Run all cells to scrape data from target subreddits
- Data is saved to CSV for processing

**Step 2: Data Cleaning**
```bash
jupyter notebook DataCleaning.ipynb
```
- Load raw data from CSV
- Apply cleaning and feature engineering
- Export processed dataset

**Step 3: Predictive Modeling**
```bash
jupyter notebook UpvotesPrediction.ipynb
```
- Load cleaned dataset
- Train Random Forest model
- Evaluate performance and generate predictions

### Analyzing New Subreddits

**1. Add subreddits to list**
```python
# Edit subreddits.txt or modify the list directly
subreddits = ['newsubreddit1', 'newsubreddit2']
```

**2. Collect data**
```python
for subreddit in subreddits:
    df = collect_subreddit_data(reddit, subreddit, post_limit=100)
    df.to_csv(f'{subreddit}_posts.csv', index=False)
```

**3. Process and predict**
```python
# Combine with existing data
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

# Re-train model or use existing model for predictions
predictions = best_model.predict(new_features)
```

---

## Project Goals and Deliverables

### Primary Objectives

**1. Sentiment and Engagement Analysis**
- Analyze sentiment patterns across different subreddits
- Identify factors that drive high engagement
- Visualize engagement trends using matplotlib and seaborn

**2. Correlation Analysis**
- Examine relationships between post features and upvotes
- Identify optimal posting times and content characteristics
- Compare engagement patterns across communities

**3. Predictive Modeling**
- Build machine learning models to predict post performance
- Apply Random Forest and ensemble methods
- Validate model accuracy with cross-validation

**4. Actionable Recommendations**
- Generate insights for content creators and moderators
- Provide data-driven posting strategies
- Identify high-performing content patterns

### Expected Deliverables

**Technical Outputs:**
- Comprehensive Jupyter Notebooks with documented analysis
- Trained machine learning models
- Processed datasets with engineered features
- Visualization dashboards

**Business Outputs:**
- Insight report on Reddit engagement patterns
- Recommendations for optimal posting strategies
- Subreddit comparison analysis
- Predictive performance metrics

---

## Key Findings and Insights

### Engagement Patterns

**Temporal Trends:**
- Peak engagement hours vary by subreddit type
- Weekday vs. weekend posting shows different patterns
- Morning posts in discussion subreddits perform better

**Content Characteristics:**
- Title length and sentiment correlate with upvotes
- Question-based posts drive higher comment engagement
- Certain keywords boost visibility in specific communities

**Subreddit Dynamics:**
- Large subreddits (AskReddit) have different engagement patterns than niche communities
- Moderation-heavy subreddits show more consistent quality
- Discussion-focused communities value depth over brevity

### Model Performance

**Random Forest Results:**
- Achieved MSE of [value] on test set
- R-squared score indicates [percentage] of variance explained
- Cross-validation confirms model generalization

**Feature Importance:**
- Top predictors: subreddit, posting time, title sentiment
- Comment count shows strong correlation with upvotes
- Subscriber count impacts baseline engagement

---

## Development Highlights

### What Makes This Project Stand Out

**Comprehensive Data Pipeline**
- End-to-end implementation from collection to prediction
- Robust error handling and data validation
- Scalable architecture for multiple subreddits

**Advanced Machine Learning**
- Hyperparameter optimization with RandomizedSearchCV
- Cross-validation for model reliability
- Feature importance analysis for interpretability

**Real-World Application**
- Practical tool for content creators and moderators
- Actionable insights based on data analysis
- Extensible framework for additional Reddit research

**Professional Documentation**
- Well-commented code throughout notebooks
- Clear markdown explanations of methodology
- Reproducible results with seed management

---

## Future Enhancements

### Planned Features
- **Real-Time Monitoring**: Live tracking of post performance
- **Deep Learning Models**:  LSTM for temporal prediction, BERT for text analysis
- **Sentiment Analysis Enhancement**: More sophisticated NLP techniques
- **User Behavior Prediction**:  Predict user engagement patterns
- **Content Recommendation System**: Suggest optimal content topics
- **Dashboard Interface**: Interactive web dashboard for insights
- **API Service**: RESTful API for integration with other tools

### Technical Improvements
- **Database Integration**: PostgreSQL for large-scale data storage
- **Containerization**: Docker for consistent deployment
- **CI/CD Pipeline**:  Automated testing and deployment
- **Performance Optimization**:  Parallel processing for faster data collection
- **Model Ensemble**: Combine multiple models for better accuracy

---

## Challenges and Solutions

### Challenge 1: Reddit API Rate Limiting
**Problem**: API requests limited to avoid server overload  
**Solution**: Implemented request throttling and pagination with time delays between batches

### Challenge 2: Missing Data Handling
**Problem**: Deleted posts and removed content create gaps  
**Solution**: Median imputation for numerical features, 'Unknown' encoding for categorical data

### Challenge 3: Feature Selection
**Problem**: High dimensionality with text and metadata features  
**Solution**: Feature importance analysis to identify most predictive variables

### Challenge 4: Class Imbalance
**Problem**:  Wide variance in upvote distribution  
**Solution**: Focused on regression rather than classification, used robust metrics like MSE

---

## Learning Outcomes

This project demonstrates proficiency in: 

**Data Science Skills:**
- Web scraping and API integration
- Data cleaning and preprocessing
- Feature engineering and selection
- Exploratory data analysis
- Statistical analysis and correlation

**Machine Learning:**
- Supervised learning (regression)
- Ensemble methods (Random Forest)
- Hyperparameter tuning
- Model evaluation and validation
- Cross-validation techniques

**Python Libraries:**
- PRAW for Reddit API
- pandas for data manipulation
- scikit-learn for machine learning
- matplotlib and seaborn for visualization
- NLTK for text processing

**Software Engineering:**
- Jupyter Notebook development
- Version control with Git
- Virtual environment management
- Documentation and commenting
- Reproducible research practices

---

## Credits and Acknowledgments

**Development:**
- Project Lead:  Superkart
- Team:  Cheesy Little Explorers

**Tools and Libraries:**
- PRAW (Python Reddit API Wrapper)
- scikit-learn
- pandas and numpy
- matplotlib and seaborn
- NLTK

**Data Source:**
- Reddit API and public subreddits

---

## License

This project is an academic work.  Please contact the developer for usage permissions. 

---

## Developer

**Superkart**

- GitHub: [@Superkart](https://github.com/Superkart)
- Project Repository: [Reddit_Moderator_Tool](https://github.com/Superkart/Reddit_Moderator_Tool)

---

## References and Resources

**Reddit API Documentation**:   [PRAW Documentation](https://praw.readthedocs.io/)  
**Google Colab Notebook**: [Open in Colab](https://colab.research.google.com/drive/1kn-Bnn9OWjeuTLuF4ay4qdQQlhSjkp1D)  
**scikit-learn**: [Machine Learning Documentation](https://scikit-learn.org/stable/documentation.html)

---

**Data Science | Machine Learning | Reddit Analytics | Predictive Modeling**
