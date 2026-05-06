import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 📂 Load data
df = pd.read_csv("students.csv")

# 📊 Compute average
df["average"] = df[["math", "english", "science"]].mean(axis=1)

# =========================
# 📈 1. Bar Chart - Student Averages
# =========================
plt.figure()

plt.bar(df["name"], df["average"])
plt.title("Student Average Scores")
plt.xlabel("Students")
plt.ylabel("Average Score")

plt.show()


# =========================
# 📉 2. Line Chart - Subject Comparison
# =========================
plt.figure()

plt.plot(df["name"], df["math"], label="Math")
plt.plot(df["name"], df["english"], label="English")
plt.plot(df["name"], df["science"], label="Science")

plt.title("Subject Performance Comparison")
plt.xlabel("Students")
plt.ylabel("Scores")
plt.legend()

plt.show()


# =========================
# 🥧 3. Pie Chart - Class Performance Distribution
# =========================
plt.figure()

labels = df["name"]
sizes = df["average"]

plt.pie(sizes, labels=labels, autopct="%1.1f%%")
plt.title("Class Contribution to Total Performance")

plt.show()