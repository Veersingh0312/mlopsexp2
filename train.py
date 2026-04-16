from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib

X, y = load_iris(return_X_y=True)
model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model trained and saved!")