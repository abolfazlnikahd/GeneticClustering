import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# met the dataset
dbs_df = pd.read_excel("Dry_Bean_Dataset.xlsx")
for row in dbs_df.head().values:
    print(row)
"""
# label encoding for Class column
le = LabelEncoder()
dbs_df['Class_code'] = le.fit_transform(dbs_df['Class'])

# complete the missing data with median
columns = dbs_df.drop("Class", axis=1).columns
for column in columns:
    dbs_df[column] = dbs_df[column].fillna(dbs_df[column].median())

# The following code snippet is for finding outliers
###cluster = KMeans(n_clusters=7)
cluster.fit(dbs_df.drop("Class", axis=1))

plt.scatter(dbs_df["Area"], dbs_df["Perimeter"], c=cluster.labels_)
plt.show()###

# the followiing code sinppet is for saving cleaned data to new dataset
dbs_df = dbs_df.drop("Class", axis=1)

#dbs_df.to_excel("cleaned_data.xlsx")


"""