
# do data loading
from pathlib import Path 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans

classification_all = pd.read_csv(Path(f'/Users/hhudson/Downloads/classification_data_all.csv'))
subgrouper = KMeans(n_clusters=5)
pca_sub = PCA(n_components=2)
pre_data = classification_all.iloc[:, 1:]
pre_cancer = classification_all.iloc[:, 0]
pca_sub.fit_transform(pre_data)
pca_data = pca_sub.components_
cancer_df1 = pd.DataFrame(pca_data.transpose())
cancer_df1.insert(0, 'cancer', pre_cancer)
cancer_grouped = cancer_df1.groupby(['cancer'])
One_cancer_type = pd.DataFrame(cancer_grouped.get_group('PRAD'))
One_cancer_type_data = One_cancer_type.iloc[:, 1:]
pred = subgrouper.fit_predict(One_cancer_type_data)
One_cancer_type.insert(0, 'cluster', pred)
sns.scatterplot(data=One_cancer_type, x=One_cancer_type.iloc[:, 2], y=One_cancer_type.iloc[:, 3], hue='cluster')