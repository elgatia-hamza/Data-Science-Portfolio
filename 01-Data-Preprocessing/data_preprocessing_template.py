# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,3].values


# Data preprocessing
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])



# Encoding the Categorical Variable
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=1/4, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain[:, 3:] = sc.fit_transform(Xtrain[:, 3:])
Xtest[:, 3:] = sc.transform(Xtest[:, 3:])
