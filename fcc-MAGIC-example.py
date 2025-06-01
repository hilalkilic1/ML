import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df.head()                           # Displays the first few rows of the DataFrame
#print(df.head())

#print(df['class'].unique())          Shows different values in the class column 


df['class'] = (df['class'] == 'g').astype(int)
#print(df['class'].unique())

# for label in cols[:-1]:
#     plt.hist(df[df['class'] == 1][label], color="blue", alpha=0.7, density= True)
#     plt.hist(df[df['class'] == 0][label], color="red", alpha=0.7, density= True)
#     plt.title(label)
#     plt.xlabel(label)
#     plt.ylabel("Probability")
#     plt.legend()
#     plt.show()  

# TRAIN, VALIDATION, TEST SPLIT

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_datasets(dataframe, oversample = False): 
    X = dataframe[cols[:-1]].values
    Y = dataframe[cols[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample: 
        ros = RandomOverSampler()
        X, Y = ros.fit_resample(X, Y)

    data = np.hstack((X, np.reshape(Y, (-1, 1))))

    return data, X, Y
    
'''
print(len(train[train["class"]==1]))
print(len(train[train["class"]==0]))
'''
train, X_train, Y_train = scale_datasets(train, oversample=True)
valid, X_valid, Y_valid = scale_datasets(valid, oversample=False)
test, X_test, Y_test = scale_datasets(test, oversample=False) 

#kNN

#  k-nearest neighbors (k is the 
#                       number of neighbors to consider, 
#                       with the euclidean distance)


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)

Y_pred = knn_model.predict(X_test) 

#print(classification_report(Y_test, Y_pred))

#Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, Y_train)

Y_pred_nb = nb_model.predict(X_test)
#print(classification_report(Y_test, Y_pred_nb))


#Logistic Regression p = S(mx + b)

#Sigmoid Function S(y) = 1 / (1 + e^(-y))

from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression()
lg_model = nb_model.fit(X_train, Y_train)

#Support Vector Machine
#kernel trick 

from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train, Y_train)

Y_pred_svm = svm_model.predict(X_test)
print(classification_report(Y_test, Y_pred_svm))

#Neural Network

