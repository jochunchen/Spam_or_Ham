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

5. Visualization:
   - ROC Curve: A Receiver Operating Characteristic (ROC) curve is plotted to show the trade-off between the true positive rate and the false positive rate. The Area Under the Curve (AUC) score is also calculated and displayed, indicating the model's overall ability to distinguish between classes.

   - Confusion Matrix: A confusion matrix is plotted to visualize the number of correct and incorrect predictions. This helps to understand which classes the model is confusing.
  
6. Spam Content Analysis:
   - The script identifies the emails that were classified as spam by the model.
   - It then uses `CountVectorizer` to count the word frequencies specifically within these spam emails.
   - A horizontal bar chart is generated showing the top 50 most frequent words, providing insight into the common language and themes of spam emails.
  
7. K-Means Clustering:
   - Elbow Method: The Elbow Method is used to determine the optimal number of clusters (`k`) for K-Means. It plots the inertia (within-cluster sum of squares) against the number of clusters. The "elbow" point on the graph suggests the best `k`.
   - Clustering: The K-Means algorithm is applied to the spam email data to group them into distinct clusters based on their word frequencies. A scatter plot of the clusters is generated.

8. TF-IDF Analysis of Clusters:
   - TF-IDF Vectorization: `TfidfVectorizer` is used to compute the Term Frequency-Inverse Document Frequency (TF-IDF) score, which reflects the importance of a word in a document relative to the entire dataset.
   - Cluster-Specific Word Analysis: A function is used to find the most representative words for each cluster based on their average TF-IDF scores.
   - Cluster Visualization: Bar plots are created for each cluster, visualizing the top 15 words and their scores. This helps in understanding the distinct themes of different spam email groups (e.g., finance-related spam, health-related spam, etc.).
