import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import pickle

# Load your CSV data
data = pd.read_csv('table-data.csv', encoding='ISO-8859-1')

# Assuming the last column contains the class labels, and others are features
X = data.iloc[:, :-1]  # All columns except the last one are features
y = data.iloc[:, -1]  # Last column is the target variable (class)

# Combine all feature columns into a single string per row
X_combined = X.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Initialize the CountVectorizer
#vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()


# Fit and transform the feature data
X_vectorized = vectorizer.fit_transform(X_combined)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
class_weights = compute_class_weight('balanced', classes=y.unique(), y=y)
clf = MultinomialNB(class_prior=class_weights)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained classifier
with open('classifier.pickle', 'wb') as clf_file:
    pickle.dump(clf, clf_file)

# Save the vectorizer
with open('vectorizer.pickle', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)