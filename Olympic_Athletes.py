import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

file_path = 'Olympic_Athletes.csv'
data = pd.read_csv("Olympic_Athletes.csv")

vectorizer = CountVectorizer()
X_disciplines = vectorizer.fit_transform(data['disciplines'].fillna(''))

data['birth_date'] = pd.to_datetime(data['birth_date'], format='%d-%m-%Y', errors='coerce', dayfirst=True)
data['birth_date'] = data['birth_date'].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
X_birth_date = data[['birth_date']].fillna(0).values

X = np.hstack([X_disciplines.toarray(), X_birth_date])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

data['cluster'] = clusters

data.to_csv(file_path, index=False)

print("Clustering completed and results saved to the CSV file.")
