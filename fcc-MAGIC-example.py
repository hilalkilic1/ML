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
    
print(len(train[train["class"]==1]))
print(len(train[train["class"]==0]))