from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


na_vals = ['NAN', 'N/A', np.nan, '?']
data = pd.read_csv('breast-cancer-wisconsin.csv', na_values=na_vals, names=['Sample code number', 'Clump Thickness', ' Uniformity of Cell Size', 'Uniformity of Cell Shape',
                                                                            'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                                                                            'Normal Nucleoli', 'Mitoses', 'Class'])

# 1- Check the dataset if it needs a clean or not

data.isnull().sum()
data.dtypes
data_output = data['Class']
data_input = data.iloc[:, 1:-1].values


# 2- Clean dataset by filling the missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="median")
imputer = imputer.fit(data_input[:, 5:6])
data_input[:, 5:6] = imputer.transform(data_input[:, 5:6])

data_input = pd.DataFrame(data_input)
data_input = data_input.values.astype(int)


# Give columns names
data_input = pd.DataFrame(data_input, columns=['Clump Thickness', ' Uniformity of Cell Size', 'Uniformity of Cell Shape',
                                               'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                                               'Normal Nucleoli', 'Mitoses'])

data_input.dtypes  # all the datatypes are int

# 3- Training the Decision Tree Model

# Here random state to fix the same random values
X, X_test, y, y_test = train_test_split(
    data_input, data_output, test_size=0.33, random_state=2)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.33, random_state=2)

classifier = DecisionTreeClassifier(criterion='entropy', random_state=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
# I give him X_train to predict y_pred_train
y_pred_train = classifier.predict(X_train)
y_pred_val = classifier.predict(X_val)
# Check accuracy of trainnig
print(accuracy_score(y_train, y_pred_train))
print(accuracy_score(y_val, y_pred_val))

# Check accuracy of testing
y_pred_test = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred_test))


# 4- The number of rows that are predicted wrong can view it with Confusion Matrix

print(confusion_matrix(y_test, y_pred_test))
############ Please Uncomment this after running the whole project ##################

# matrix= plot_confusion_matrix(classifier , X_test , y_test , cmap = plt.cm.Reds)
# matrix.ax_.set_title('Confusion Matrix' , color='White')


# 6- Descion tree visualisation
plt.figure(figsize=(150, 128))
tree.plot_tree(classifier, feature_names=['Clump Thickness', ' Uniformity of Cell Size', 'Uniformity of Cell Shape',
                                          'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                                          'Normal Nucleoli', 'Mitoses'], class_names=['bengin', 'malignant'], filled=True)
