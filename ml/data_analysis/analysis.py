import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set(style="whitegrid")

CURRENT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = CURRENT_DIR / "data" / "raw" / "data-final.csv"


df = pd.read_csv(DATA_PATH, sep="\t")

print("\nShape:", df.shape)
print("Columns sample:", df.columns[:10])

EXT = [f"EXT{i}" for i in range(1, 11)]
EST = [f"EST{i}" for i in range(1, 11)]
AGR = [f"AGR{i}" for i in range(1, 11)]
CSN = [f"CSN{i}" for i in range(1, 11)]
OPN = [f"OPN{i}" for i in range(1, 11)]

df["Extraversion"] = df[EXT].mean(axis=1)
df["Neuroticism"] = df[EST].mean(axis=1)
df["Agreeableness"] = df[AGR].mean(axis=1)
df["Conscientiousness"] = df[CSN].mean(axis=1)
df["Openness"] = df[OPN].mean(axis=1)

traits = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]


print("\n=== TRAIT STATISTICS ===")
print(df[traits].describe())


for trait in traits:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[trait], kde=True)
    plt.title(f"Distribution of {trait}")
    plt.xlabel(trait)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

for trait in traits:
    plt.figure(figsize=(4, 5))
    sns.boxplot(y=df[trait])
    plt.title(f"Outliers in {trait}")
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(
    df[traits].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Big Five Traits")
plt.tight_layout()
plt.show()
