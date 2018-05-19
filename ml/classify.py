import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
import pickle

print(sys.argv)
if len(sys.argv) < 3:
    print ("Please provide the path of the tsv file an diretory to store the data model for prediction.")
    print ('Example: python classify.py "AP_ICD10.tsv" "../state/"')
    sys.exit(1)

# Pickle file paths
MDL_PKL_FILENAME = os.path.join(sys.argv[2], 'svc_model.sav')
VCT_PKL_FILENAME = os.path.join(sys.argv[2], 'tfidf_vect.sav')

all_df = pd.read_csv(sys.argv[1], sep ='\t', delimiter='\t', header=0)
col = ['icd10encounterdiagcode', 'A/P'] # target, feature
df = all_df[col]
df = df[pd.notnull(df['A/P'])]
df.columns = ['icd10encounterdiagcode', 'A/P']
df['icd_id'] = df['icd10encounterdiagcode'].str.slice(0,3)
icd_id_df = df[['icd10encounterdiagcode', 'icd_id']].drop_duplicates().sort_values('icd_id')
icd_to_id = dict(icd_id_df.values)
id_to_category = dict(icd_id_df[['icd_id', 'icd10encounterdiagcode']].values)

tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df['A/P']).toarray()
labels = df.icd_id
print(features.shape)

model = OneVsRestClassifier(LinearSVC())

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(metrics.classification_report(y_test, y_pred, target_names=df['icd_id'].unique()))

# save the model to disk
pickle.dump(tfidf, open(VCT_PKL_FILENAME, 'wb'))
pickle.dump(model, open(MDL_PKL_FILENAME, 'wb'))
