import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import Center_Points
import Constant
import joblib

def train_model():
    # Assuming you have imported CentralDots and Constant properly

    # Load your data
    myDots = Center_Points.Center_Register_Signatures()
    all_x_values = myDots[0]
    all_y_values = myDots[1]
    X = np.column_stack((all_x_values, all_y_values))

    # Data preprocessing: Scale and center the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create a One-Class SVM model
    model = svm.OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')  # You can adjust 'nu' and kernel parameters

    # Fit the model on the scaled data
    model.fit(X_scaled)

    # Generate a fine-grained grid for visualization
    x_min, x_max = X_scaled[:, 0].min() - 0.1, X_scaled[:, 0].max() + 0.1
    y_min, y_max = X_scaled[:, 1].min() - 0.1, X_scaled[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    joblib.dump((model, scaler, xx, yy, Z), 'trained_model_svm.pkl')
    print("model saved")

def load_model():
    # Load the trained model
    model, scaler, xx, yy, Z = joblib.load('trained_model_svm.pkl')
    print("model is load")

    # Load test data
    csv_login = Constant.SERVER_PATH + r"\login.csv"
    data = np.loadtxt(csv_login, delimiter=',')
    x_values = data[:, 0]
    y_values = data[:, 1]
    test_data = Center_Points.Center_Array_Point_On_Board([x_values, y_values])

    # Scale test data using the loaded scaler
    test_data_scaled = scaler.transform(test_data)

    # Make predictions on the scaled test data
    predictions = model.predict(test_data_scaled)

    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], cmap=plt.cm.PuBu, alpha=0.8)
    # Note: Replace X_scaled with test_data_scaled
    plt.scatter(test_data_scaled[:, 0], test_data_scaled[:, 1], color='red', label='LOGIN Anomalies')
    plt.title('One-Class SVM Decision Boundary')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.gca().invert_yaxis()
    plt.show()

    threshold = 0.75 * len(predictions)
    counter = (predictions == 1).sum()

    if counter >= threshold:
        return True
    else:
        return False

#train_model()
#load_model()


