import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("survey.csv")

df = df.drop(columns=['comments','Timestamp'])

name_of_features = df.columns.to_list()

df = df[df['Age']>=18]
df = df[df['Age']<=100]
df['Age'].value_counts()

df_encoded = pd.DataFrame()
encoder = LabelEncoder()
for feature in name_of_features:
    df_encoded[feature] = encoder.fit_transform(df[feature])

for feature in name_of_features:
    vars = [name for name in name_of_features if name!=feature]
    sns.pairplot(
    df_encoded,
    hue = feature,
    vars = vars,
    palette='Set1'
    )
    plt.suptitle("Pairwise Relationships: Age, Fare, and Class", y=1.02)
    plt.tight_layout()
    print(plt.show)

print(9)