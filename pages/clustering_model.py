import streamlit as st
import matplotlib.pyplot as plt
st.title("UNSUPERVISED LEARNING")
st.header(":blue[KMeans Clustering Algorithm]",divider='blue')
st.write("___K-Means Clustering is an unsupervised machine learning algorithm used to group data into K distinct clusters based on similarity. It works by randomly initializing K cluster centroids, assigning each data point to the nearest centroid, and then updating the centroids based on the mean of the assigned points.___")
st.write('This process repeats until the assignments no longer change. K-Means is simple, fast, and effective for finding structure in unlabeled data, especially when clusters are well-separated.')
st.header(":blue[OUTPUT GRAPH OF CLUSTERING]",divider='blue')
st.image("output.png")
st.subheader(":blue[metrics]")
st.write('silhouette_score of the model is 0.35')