import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap

# Load the model
MODEL_PATH = "models/RandomForestClassifier.sav"
loaded_model = pickle.load(open(MODEL_PATH, 'rb'))

# Load data
data = pd.read_csv('checking2.csv')

def find_PQRST_ECG(a):
    for i in range(22, len(a) - 22):
        a[i] = (a[i] + a[i - 1] + a[i + 1] + a[i - 2] + a[i + 2] + a[i - 3] +
                a[i + 3] + a[i - 4] + a[i + 4] + a[i - 5] + a[i + 5] + a[i - 6] +
                a[i + 6] + a[i - 7] + a[i + 7] + a[i - 8] + a[i + 8]) / 17
    a = list(a)
    b = a.copy()
    b.sort()
    m = b[len(b) // 2]
    max_ = max(a)
    min_ = min(a)
    temp_ = []
    Q, Qi, R, Ri, P, Pi, S, Si, T, Ti = None, None, None, None, None, None, None, None, None, None
    for i in range(8, len(a) - 8):
        if a[i] > a[i - 2] and a[i] > a[i + 2]:
            R = a[i]
            temp_.append(R)
            R = max(temp_)
            Ri = a.index(R)
    temp_.clear()
    for i in range(Ri, 0, -1):
        if a[i] < a[i - 2] and a[i] < a[i + 2] and i < Ri:  # a[i]<(min_+0.2) and
            Q = a[i]
            temp_.append(Q)
            Q = min(temp_)
            Qi = a.index(Q)
    if Q is None:
        Q = a[Ri - 10]
        Qi = Ri - 10
    temp_.clear()
    for i in range(Qi, 0, -1):
        if a[i] > a[i - 2] and a[i] > a[i + 2]:  # a[i]>max_*0.2 and
            P = a[i]
            temp_.append(P)
            P = max(temp_)
            Pi = a.index(P)
    if P is None:
        P = a[Qi - 10]
        Pi = Qi - 10
    temp_.clear()
    for i in range(Ri, len(a) - 2, 1):
        if a[i] < a[i - 2] and a[i] < a[i + 2]:  # a[i]>max_*0.2 and
            S = a[i]
            temp_.append(S)
            S = min(temp_)
            Si = a.index(S)
    temp_.clear()
    if S is not None:
        for i in range(Si, len(a) - 22, 1):
            if a[i] > a[i - 2] and a[i] > a[i + 2]:  # a[i]>max_*0.2 and
                T = a[i]
                temp_.append(T)
                T = max(temp_)
                Ti = a.index(T)
    else:
        print('S not found!!!!')
        try:
            S = a[Ri + 10]
            Si = Ri + 10
        except:
            S = a[Ri + 1]
            Si = Ri + 1
    if T is None:
        print('T not found!!!!')
        try:
            Ti = Si + 4
            T = a[Si + 4]
        except:
            Ti = Si
            T = a[Si]
    PQi = range(Pi, Qi, 1)
    QRi = range(Qi, Ri, 1)
    RSi = range(Ri, Si, 1)
    STi = range(Si, Ti, 1)
    plt.plot(a)
    plt.plot(Pi, P, "ob", label="P")
    plt.plot(Qi, Q, "og", label="Q")
    plt.plot(Ri, R, "or", label="R")
    plt.plot(Si, S, "oy", label="S")
    plt.plot(Ti, T, "oc", label="T")
    plt.plot(PQi, np.ones(len(PQi)) * Q, "--g", label="PQ")
    x = np.array([Pi, Pi])
    y = np.array([P, Q])
    plt.plot(x, y, '--k')
    plt.plot(QRi, np.ones(len(QRi)) * R, "--r", label="QR")
    x = np.array([Qi, Qi])
    y = np.array([Q, R])
    plt.plot(x, y, '--k')
    plt.plot(RSi, np.ones(len(RSi)) * S, "--y", label="RS")
    x = np.array([Ri, Ri])
    y = np.array([R, S])
    plt.plot(x, y, '--k')
    plt.plot(STi, np.ones(len(STi)) * T, "--c", label="ST")
    x = np.array([Si, Si])
    y = np.array([S, T])
    plt.plot(x, y, '--k')
    plt.legend()
    plt.title('ECG')
    plt.xlabel('Time')
    plt.savefig('ecg.png', bbox_inches='tight', dpi=300)
    plt.close()
    return m, R, Ri, Q, Qi, P, Pi, S, Si, T, Ti, QRi, PQi, RSi, STi

def testing_in_data(data, sample_number):
    m, R, Ri, Q, Qi, P, Pi, S, Si, T, Ti, QRi, PQi, RSi, STi = find_PQRST_ECG(data.iloc[sample_number, :-1].values)
    X = np.asanyarray([m, R, Ri, Q, Qi, P, Pi, S, Si, T, Ti])
    result = loaded_model.predict(np.array([X]))
    return result[0]

class ECGCheckerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(ECGCheckerApp, self).__init__()
        uic.loadUi("ecg_checker.ui", self)

        # Load the logo image
        pixmap = QPixmap("logo.png")
        self.logo.setPixmap(pixmap)

        # Connect buttons to functions
        self.checkButton.clicked.connect(self.check_sample)
        self.resetButton.clicked.connect(self.reset_form)

    def check_sample(self):
        try:
            sample_number = int(self.sampleInput.text())
            if sample_number < 1 or sample_number > 1880:
                raise ValueError("Sample number out of range.")
            
            result = testing_in_data(data, sample_number - 1)

            # Update the ECG image
            pixmap = QPixmap("ecg.png")
            self.ecgImage.setPixmap(pixmap)

            # Show the result
            self.resultLabel.setText(str(result))
        except Exception as e:
            print(f"Error: {e}")
            self.resultLabel.setText("Err")

    def reset_form(self):
        self.sampleInput.clear()
        self.resultLabel.clear()
        self.ecgImage.clear()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ECGCheckerApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
