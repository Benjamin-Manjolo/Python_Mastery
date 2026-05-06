import pandas as pd
import numpy as np

#load data
df = pd.read_csv('students.csv')

#calculate avaerage score per student
df["average"] = df[["math","english","science"]].mean(axis=1)

print("\nAverage score per student:")
print(df)

#find top student
top_student = df.loc[df["average"].idxmax()]

print("\nTop student:")
print(top_student["name"],"->",top_student["average"])

#find failing students (below 50 avaerage)
failing = df[df["average"] < 50]

print("\n❌ Failing Students:")
print(failing[["name", "average"]])

# 📈 Class statistics
print("\n📈 Class Stats:")
print("Math avg:", np.mean(df["math"]))
print("English avg:", np.mean(df["english"]))
print("Science avg:", np.mean(df["science"]))
print("Overall avg:", np.mean(df["average"]))