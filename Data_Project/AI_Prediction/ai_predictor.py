import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 📂 Load Data
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "students.csv")
df = pd.read_csv(file_path)

# 🎯 Features (input)
X = df[["math", "english", "science"]]

# 🎯 Target (output)
y = df["pass"]

# ✂️ Split data (train/test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 🤖 Create model
model = LogisticRegression()

# 🧠 Train model
model.fit(X_train, y_train)

# 📊 Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# 🔮 Predict new student
new_student = [[85, 90, 92]]  # math, english, science

# ⚠️ THIS WAS MISSING (IMPORTANT)
prediction = model.predict(new_student)

# 🧾 Output result
if prediction[0] == 1:
    print("Prediction: PASS 🎉")
else:
    print("Prediction: FAIL ❌")