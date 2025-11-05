import pandas as pd
from sklearn.cluster import KMeans

# Baca data CSV
data = pd.read_csv('ponpes_final.csv')  # Ganti dengan nama file CSV kamu

# Pisahkan depot dan customer
depot = data[data['id'] == 0]
customers = data[data['id'] != 0].copy()  # Cluster hanya customer

# Koordinat untuk clustering
coords = customers[['latitude', 'longitude']].values
kmeans = KMeans(n_clusters=7, random_state=42)
labels = kmeans.fit_predict(coords)
customers['cluster'] = labels

# Output tiap cluster ke file CSV
total_cluster = 7
for i in range(total_cluster):
    in_cluster = customers[customers['cluster'] == i]
    cluster_csv = pd.concat([depot, in_cluster], ignore_index=True)
    cluster_csv.to_csv(f'GIB_cluster{i+1}.csv', index=False)
print('Clustering done. File GIB_cluster1..7.csv sudah dibuat.')
