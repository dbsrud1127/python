#
import pandas as pd

df = pd.read_csv("train.csv")

#
print(df.head())

#
df.info()

#
print("Age 평균:", df["Age"].mean())
print("Age 최솟값:", df["Age"].min())
print("Age 최대값:", df["Age"].max())

print("Fare 평균:", df["Fare"].mean())
print("Fare 최솟값:", df["Fare"].min())
print("Fare 최대값:", df["Fare"].max())

print(df[["Age", "Fare"]].agg(["mean","min","max"]))

#
survived = (df["Survived"] == 1).sum()
dead = (df["Survived"] == 0).sum()

print("생존자:", survived)
print("사망자:", dead)

#
print(df.groupby("Pclass").size())

#
df_50 = df[df["Age"] >= 50]

print(df_50.shape)
print(df_50.head())

#









