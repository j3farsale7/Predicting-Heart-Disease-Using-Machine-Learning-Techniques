import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import os




data1 = pd.read_csv('data.csv')
data1.fillna(970.500000, inplace=True)  # Corrected inplace parameter
print(data1.head())


signal = data1.iloc[0, :-1].values
plt.plot(signal)
plt.show()


max_value = max(signal)
max_index = list(signal).index(max_value)
print(max_index, max_value)


plt.plot(signal)
plt.plot(max_index, max_value, "or", label="max_value")
plt.legend()
plt.show()


temp = []
tempi = []
for i in range(1, len(signal) - 1):  # Avoid index out of range error
    if signal[i] > max_value * 0.9 and signal[i] > signal[i + 1] and signal[i] > signal[i - 1]:
        temp.append(signal[i])
        tempi.append(i)

plt.plot(signal)
plt.plot(tempi, temp, "og", label="local_maxima")
plt.legend()
plt.show()



data_temp = []
for j in range(len(data1)):
    signal = data1.iloc[j, :-1].values
    max_value = max(signal)
    c = 1
    print(f'Converting signal {j+1} to samples:')
    for i in range(5, len(signal) - 5):
        if signal[i] > max_value * 0.9 and signal[i] > signal[i + 1] and signal[i] > signal[i - 1]:
            if 1 < c < 6:
                print(f'#{c-1}')
                sample = signal[max(0, i - 100):min(len(signal), i + 200)]
                row = pd.DataFrame([sample])
                row['label'] = data1.iloc[j, -1]
                data_temp.append(row)
            c += 1


data = pd.concat(data_temp, axis=0)
data.to_csv('data_2.csv', index=False)
data2 = pd.read_csv('data_2.csv')
print(data2.head())
print(len(data2))


plt.plot(data2.iloc[0, :-1].values)
plt.show()




def find_PQRST_data_ECG(a):
    for i in range(22, len(a) - 22):
        a[i] = sum([a[i + j] for j in range(-8, 9)])/17

    a = list(a)
    b = sorted(a)
    m = b[len(b) // 2]
    max_value = max(a)
    min_value = min(a)
    temp_ = []
    Q, Qi, R, Ri, P, Pi, S, Si, T, Ti = [None] * 10

    # Find R peak
    for i in range(8, len(a) - 8):
        if a[i] > a[i - 2] and a[i] > a[i + 2]:
            R = a[i]
            temp_.append(R)
    R = max(temp_)
    Ri = a.index(R)
    temp_.clear()

    # Find Q point
    for i in range(Ri, 0, -1):
        if a[i] < a[i - 2] and a[i] < a[i + 2]:
            Q = a[i]
            temp_.append(Q)
    Q = min(temp_) if temp_ else a[Ri - 10]
    Qi = a.index(Q) if temp_ else Ri - 10
    temp_.clear()

    # Find P point
    for i in range(Qi, 0, -1):
        if a[i] > a[i - 2] and a[i] > a[i + 2]:
            P = a[i]
            temp_.append(P)
    P = max(temp_) if temp_ else a[Qi - 10]
    Pi = a.index(P) if temp_ else Qi - 10
    temp_.clear()

    # Find S point
    for i in range(Ri, len(a) - 2):
        if a[i] < a[i - 2] and a[i] < a[i + 2]:
            S = a[i]
            temp_.append(S)
    
    # Ensure index is within range
    if temp_:
        S = min(temp_)
        Si = a.index(S)
    else:
        if Ri + 10 < len(a):
            S = a[Ri + 10]
            Si = Ri + 10
        else:
            S = a[-1]  # Use the last element if out of range
            Si = len(a) - 1
    temp_.clear()

    # Find T point
    if S is not None:
        for i in range(Si, len(a) - 22):
            if a[i] > a[i - 2] and a[i] > a[i + 2]:
                T = a[i]
                temp_.append(T)
    
    # Ensure index is within range
    if temp_:
        T = max(temp_)
        Ti = a.index(T)
    else:
        if Si + 4 < len(a):
            T = a[Si + 4]
            Ti = Si + 4
        else:
            T = a[-1]  # Use the last element if out of range
            Ti = len(a) - 1

    return m, R, Ri, Q, Qi, P, Pi, S, Si, T, Ti



def make_ml_data(data):
    X = data.iloc[:, :-1].values
    Y = np.asarray(data.iloc[:, -1])
    mm = []
    for i, sample in enumerate(X):
        m, R, Ri, Q, Qi, P, Pi, S, Si, T, Ti = find_PQRST_data_ECG(sample)
        ml = pd.DataFrame({'m': [m], 'R': [R], 'Ri': [Ri], 'Q': [Q], 'Qi': [Qi], 'P': [P], 'Pi': [Pi], 'S': [S], 'Si': [Si], 'T': [T], 'Ti': [Ti], 'label': [Y[i]]})
        mm.append(ml)
        print(i, ' ::  ', ml)
    return mm

# Apply function and save results
result = make_ml_data(data2)
DT = pd.concat(result, axis=0)
DT.to_csv('data_3.csv', index=False)

# Load the dataset for training
data3 = pd.read_csv('data_3.csv')
print(data3.head())

# Split the data for training and testing
X = data3.iloc[:, :-1].values
Y = data3.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=120)



X_train = X_train.dropna()
y_train = y_train[X_train.index]



X_test = X_test.dropna()
y_test = y_test[X_test.index]


from sklearn.impute import SimpleImputer

# Create an imputer object with a chosen strategy (e.g., mean, median, most_frequent)
imputer = SimpleImputer(strategy='mean')

# Fit and transform the training data
X_train = imputer.fit_transform(X_train)

# Transform the test data (without fitting, to avoid data leakage)
X_test = imputer.transform(X_test)


from sklearn.ensemble import HistGradientBoostingClassifier

# Train and evaluate the model
clf = HistGradientBoostingClassifier()
clf.fit(X_train, y_train)
print('Model Test Accuracy:', accuracy_score(y_test, clf.predict(X_test)))
print('Model Train Accuracy:', accuracy_score(y_train, clf.predict(X_train)))



# Train and evaluate SVM model
clf = svm.SVC(kernel='rbf', gamma='auto')
clf.fit(X_train, y_train)
print('SVM Test Accuracy:', accuracy_score(y_test, clf.predict(X_test)))
print('SVM Train Accuracy:', accuracy_score(y_train, clf.predict(X_train)))



clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
print('KNN Test Accuracy:', accuracy_score(y_test, clf.predict(X_test)))
print('KNN Train Accuracy:', accuracy_score(y_train, clf.predict(X_train)))

# Train and evaluate Decision Tree model
clf = DecisionTreeClassifier(criterion='entropy', max_depth=15)
clf.fit(X_train, y_train)
print('Decision Tree Test Accuracy:', accuracy_score(y_test, clf.predict(X_test)))
print('Decision Tree Train Accuracy:', accuracy_score(y_train, clf.predict(X_train)))

# Train and evaluate Random Forest model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print('Random Forest Test Accuracy:', accuracy_score(y_test, clf.predict(X_test)))
print('Random Forest Train Accuracy:', accuracy_score(y_train, clf.predict(X_train)))

# Save the Random Forest model
if not os.path.exists('models'):
    os.makedirs('models')

MODEL_PATH = "models/RandomForestClassifier.sav"
pickle.dump(clf, open(MODEL_PATH, 'wb'))

# Load the Random Forest model
classifier = pickle.load(open(MODEL_PATH, 'rb'))

# Define a testing function for ECG data
def testing_in_data(data, sample_number):
    sample = data.iloc[sample_number, :-1].values
    m, R, Ri, Q, Qi, P, Pi, S, Si, T, Ti = find_PQRST_data_ECG(sample)
    X = np.array([m, R, Ri, Q, Qi, P, Pi, S, Si, T, Ti]).reshape(1, -1)
    prediction = classifier.predict(X)
    print(f'Sample {sample_number} Prediction:', prediction)

# Test the model with a specific sample
data = pd.read_csv('data.csv')
sample_number = 1105
testing_in_data(data, sample_number)
