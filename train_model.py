from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(data.data, data.target)

Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/model.joblib")
print("Champion model saved to model/model.joblib")
