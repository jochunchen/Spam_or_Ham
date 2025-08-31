# Spam or Ham
This project explores and compares four different machine learning models—Decision Tree, Logistic Regression, Naive Bayes, and Support Vector Machine (SVM)—for classifying spam emails. The project involves a complete pipeline from data preprocessing to model training, performance evaluation, and an in-depth analysis of the words that most contribute to spam emails.

The analysis also uses K-Means clustering to group similar spam emails and TF-IDF to identify the most representative words within each cluster, providing valuable insights into the characteristics of spam content.

## Key Features
- Spam Classification: Classifies emails as 'Spam' or 'Ham' (not spam).
- Data Preprocessing: Cleans and prepares the text data for machine learning models by removing links, special characters, and stopwords, and by performing lemmatization.
- Model Comparison: Compares the performance of four different classifiers using various metrics like accuracy, precision, recall, and F1-score.
- Performance Visualization: Generates plots for ROC curves and confusion matrices to visually represent each model's performance.
- Spam Word Analysis: Identifies and visualizes the most frequently occurring words in emails classified as spam.
- Clustering Spam Emails: Uses K-Means clustering to group spam emails and analyzes the content of each cluster using TF-IDF.

## Data
The project uses the `completeSpamAssassin.csv` dataset. (https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset?select=completeSpamAssassin.csv)



## Code Description
1. Module Imports:
   Essential libraries for data manipulation, natural language processing, machine learning, and visualization are imported.

2. Data Loading & Preprocessing:
   - The `completeSpamAssassin.csv` file is loaded into a pandas DataFrame.
   - The Unnamed: `0` column is dropped, and rows with missing values are removed.
   - The text data is cleaned by removing URLs and special characters, converting it to lowercase, and tokenizing it.
   - Lemmatization is performed using `WordNetLemmatizer` to reduce words to their base form (e.g., 'running' to 'run').
   - Stopwords (common words like 'the', 'is', 'a') are removed to focus on meaningful words.
     
3. Feature Extraction:
   `CountVectorizer` is used to convert the cleaned text data into a numerical feature matrix. This process transforms the text into a bag-of-words representation where each column corresponds to a word and each value represents its frequency.

4. Model Training and Evaluation:
   - The dataset is split into training and testing sets to evaluate the model's performance on unseen data.

   - The respective machine learning model (`DecisionTreeClassifier`, `LogisticRegression`, `GaussianNB`, or `svm.SVC`) is trained on the training data.

   - Predictions are made on the test set, and the model's performance is evaluated.

   - Accuracy and a detailed classification report (including precision, recall, and F1-score) are printed to the console.

5. 
