# ML_classification

##Steps to run the project:
1. git clone https://github.com/srijyothsna/ML_classification.git
2. cd ML_classification
3. pip install -r requirements.txt
4. python classify.py <path_to_input_file_incl_filename> <dir_path_to_save_models>
5. python predict.py <dir_path_to_save_model> <doctor_assesment_str_in_quotes>

## Framework/ Libraries
The solution is implemented in Python and tested on v3.4. The following packages (in requirements.txt) and any dependencies need to installed in the Python environment :
sklearn
scipy
pandas
numpy

## Approach
Given that the problem involves multilabel/multiclass text classification, I have used a OneVsRestClassifier with LinerSVC model, fitted on a TfIdfVectorizer. The doctor’s written assessment and plan (‘A/P'  field from the tsv file) is used as the feature vector for training. The model is trained 67% of the data, while the remaining 33% has been used as the test set.

The code is divided into 2 scripts:
1. classify.py - The data from the tsv file is read into a pandas dataframe; the classification model is built; data is divided into training/ test sets; the classifier is trained and tested on the test data; and the vectorizer and classifier are saved to file to persist the model.

This script can be run as follows: python classify.py <path_to_input_file_incl_filename> <dir_path_to_save_models>. The filenames for saving the vectorizer and classifier are hardcoded.
In this project, input files is in ‘data’ directory and models are saved into 'state’ directory.

The script expects the input file to be in tsv format similar to the sample in 'data' directory. The output from the script - classification report with precision, recall and f1 score - is printed to terminal

2. predict.py - The saved vectorizer and classifier are loaded; and runs the model on the standalone payload string.

This script can be run as follows: python predict.py <dir_path_to_save_model> <doctor_assesment_str_in_quotes>. 
In this project, models are loaded from 'state’ directory.
The output from this script - the predicted ICD code (first 3 characters) - is printed to the terminal.

The reason for splitting it into 2 scripts is to make it easier to run the model multiple times without having to train and test the model each time.

classify.py needs to be run only when there is change to the training/ test data or the classier itself. On the other hand, predict.py can be run any number of times with the currently trained model.

## Results
The overall *precision* and *recall* for this test set are *0.64* and *0.61*, respectively, with an *f1-score* of *0.61*.


## Possible improvements:
Given the limited time for solving this problem, the following steps have not been implemented, which can improve the performance further:
- NLP steps like stemming, pos tagging, named entity recognition with ICD10 as the controlled vocabulary, negation
- Use other models to train/ test like Random Forest, Bag of words or Word2vec.
- More test data and randomized training/ tests with given data to avoid overfitting/ underfitting.
