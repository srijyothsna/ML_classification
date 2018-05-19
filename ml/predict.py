import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

if len(sys.argv) < 3:
    print("Please provide the doctor A/P  as a string (in quotes) to predict the ICD code")
    print("Example: python run_model.py \"Saved state directory\" \"Doctor A/P\"")
    sys.exit()
elif len(sys.argv) >4:
    print("Please provide the doctor A/P as a string (in quotes) to predict the ICD code ")
    print("Example: python run_model.py \"Doctor A/P\"")
    sys.exit()


MDL_PKL_FILENAME = sys.argv[1] + "/" + 'svc_model.sav'
VCT_PKL_FILENAME = sys.argv[1] + "/" + 'tfidf_vect.sav'

usr_inp = sys.argv[2]
print("User input: {}\n".format(usr_inp))

# load the model from disk
loaded_model = pickle.load(open(MDL_PKL_FILENAME, 'rb'))
loaded_vect = pickle.load(open(VCT_PKL_FILENAME, 'rb'))

X_pred = loaded_vect.transform([usr_inp])
y_pred = loaded_model.predict(X_pred)

print("X={}, Predicted={}".format(usr_inp, y_pred))
