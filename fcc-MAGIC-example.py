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

import tensorflow as tf

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_ylabel('Binary Crossentropy')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.grid(True)

    plt.show()

def train_model(X_train, Y_train, num_nodes, dropout_prob, lr, batch_size, epochs=100):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(dropout_prob), #this prevetns overfitting
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    history = nn_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose = 0)
    
    return nn_model, history

least_val_loss = float('inf')
least_loss_model = None
epochs = 100
for num_nodes in [8, 16, 32, 64]:
    for dropout_prob in [0, 0.1, 0.2]:
          for lr in [0.01, 0.005, 0.001]:
            for batch_size in [16, 32, 64, 128]:
                print(f"Training with num_nodes={num_nodes}, dropout_prob={dropout_prob}, lr={lr}, batch_size={batch_size}")
                nn_model, history = train_model(X_train, Y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
                plot_history(history)
                val_loss = nn_model.evaluate(X_valid, Y_valid)[0]
                
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = nn_model



