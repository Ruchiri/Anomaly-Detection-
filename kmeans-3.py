import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import plotly.offline as plot
from scipy.spatial import distance

#read the dataset
data = pd.read_csv("dataset_stat_1.csv")
#instruments in the dataset - keep instrument id's
instruments = data['instrument_id'].unique().tolist()


#view variance in features
def view_variance_in_features():
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.show()
    return pd.DataFrame(principalComponents)

    
#plot clusters with the intertia
def view_clusters_with_inertial(PCA_components):
    inertias = []
    for k in range(1, 10):
        model = KMeans(n_clusters=k)
        model.fit(PCA_components.iloc[:,:5])
        inertias.append(model.inertia_)
    
    plt.plot(range(1, 10), inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(range(1, 10))
    plt.show()
    return inertias


#use elbow method to get cluster count
def get_cluster_count(inertias):
    acceleration = np.diff(inertias, 2)
    k = acceleration.argmax() + 3
    return k


#apply kmeans clustering and plot results
def KMeans_cluster(k, PCA_components):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(PCA_components.iloc[:,:5])
    labels = kmeans.labels_
    clusters ={}
    for i in range(k):
        cdata = PCA_components[labels == i]
        clusters[i] = cdata
    d = []
    for i in clusters:
        x = []
        y = []
        for index, row in clusters[i].iterrows():
            x.append(row[0])
            y.append(row[1])
    
        trace = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=6,
                line=dict(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5
                ),
                opacity=0.8
            )
        )
        d.append(trace)
    
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="PCA 1",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="PCA 2",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
    )
    fig = go.Figure(data=d, layout=layout)
    plot.iplot(fig, filename='simple-2d-scatter')
    return [clusters, kmeans]

#calculate the anomaly score
def calculate_scores(clusters, centroids):
    scores = []
    for i in clusters:
        tclusters = list(clusters.keys())
        tclusters = tclusters[0:i] + tclusters[i + 1:]
        dist_total = 0
        for j in tclusters:
            dist = distance.euclidean(centroids[i], centroids[j])
            dist_total += dist
        dist_mean = dist_total / len(tclusters)
        scores.append(dist_mean / len(clusters[i]))
    sqr_sum_score = 0
    for i in range(len(scores)):
        sqr_sum_score += pow(scores[i],2)
    #print cluster-->anomaly score-->normalized anomaly score    
    for i in range(len(scores)):
        stand_score = scores[i]/ pow(sqr_sum_score, 0.5)
        print("trace ",i,"-->",scores[i],"-->",stand_score)
    return scores

#get results for each instrument in the datset
for instrument in instruments:
    instrument_data = data.loc[data['instrument_id'] == instrument]
    dropped_data = instrument_data.drop(columns = ['order_id','transact_time','instrument_id'])
    #fill NAN values
    filled_data = dropped_data.fillna(method='ffill')
    filled_data = dropped_data.fillna(method='bfill')
    #standardize the features by scaling to a unit variance
    std_data = StandardScaler().fit_transform(filled_data)
    a=instrument_data.count()
    if a.instrument_id >=8:
        pca = PCA(n_components=8)
        principalComponents = pca.fit_transform(std_data)
        print("Instrument id-", instrument)
        
        PCA_components = view_variance_in_features()
        inertias = view_clusters_with_inertial(PCA_components)
        k = get_cluster_count(inertias)
        clusters, kmeans = KMeans_cluster(k, PCA_components)
        centroids = kmeans.cluster_centers_
        scores = calculate_scores(clusters, centroids)
        

    

