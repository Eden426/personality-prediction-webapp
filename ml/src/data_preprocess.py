import pandas as pd

def load_clean_dataset(data):
    col_filter = [col for col in data.columns if col[:3] in ["EXT", "EST", "AGR", "CSN", "OPN"] and not col.endswith("_E")]
    clean_dtset = data[col_filter].copy()

    reversed_to_trait = [
    "EXT2","EXT4","EXT6","EXT8","EXT10",
    "EST2","EST4",
    "AGR1","AGR3","AGR5","AGR7",
    "CSN2","CSN4","CSN6","CSN8",
    "OPN2","OPN4","OPN6"
    ]
    clean_dtset.loc[:, reversed_to_trait] = 6 - clean_dtset[reversed_to_trait]

    clean_dtset["Extraversion"] = clean_dtset[[f"EXT{i}" for i in range(1,11)]].mean(axis=1)
    clean_dtset["Neuroticism"] = clean_dtset[[f"EST{i}" for i in range(1,11)]].mean(axis=1)
    clean_dtset["Agreeableness"] = clean_dtset[[f"AGR{i}" for i in range(1,11)]].mean(axis=1)
    clean_dtset["Conscientiousness"] = clean_dtset[[f"CSN{i}" for i in range(1,11)]].mean(axis=1)
    clean_dtset["Openness"] = clean_dtset[[f"OPN{i}" for i in range(1,11)]].mean(axis=1)

    clean_dtset = clean_dtset.dropna(subset=col_filter)
    traits = ["Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness"]
    clean_dtset["DominantTrait"] = clean_dtset[traits].idxmax(axis=1)
    x = clean_dtset[col_filter]
    y = clean_dtset["DominantTrait"]


    return x, y, clean_dtset



