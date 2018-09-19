import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow
from sklearn.decomposition import PCA
from tqdm import tqdm

# load dataset by using numpy.load()
faces_image = np.load('./olivetti_faces.npy')
print('faces_image.shape: ', faces_image.shape) # 400 pictures in total, 64*64each

# set up canvas
fig, axes = plt.subplots(3, 4, figsize=(11, 6),
                         subplot_kw = {'xticks':[], 'yticks':[]},
                         gridspec_kw = dict(hspace=0.1, wspace=1.1))
print('figures in canvas: 3 rows * 4 cols, total#: ', len(axes.flat))

# show the first 12 pictures in dataset. total 12 sub-figures in 'axes.flat'
for i, ax in enumerate(axes.flat):
    ax.imshow(faces_image[i], cmap='bone')
plt.show()

'''
------------   PCA   ---------------
use PCA to do face dimensionality reduction.
faces[i]        picture of 64*64 pixels
k:              how many principle components
return:         faces_after_PCA, eigen_faces
'''
def face_pca(faces, k):
    face_count = faces.shape[0]
    face_h = faces.shape[1]
    face_w = faces.shape[2]
    
    # flatten all face images to 1-D vectors
    flatten_faces = faces.reshape(face_count, face_h*face_w)
    
    # 搞
    pca = PCA(n_components=k).fit(flatten_faces)
    
    # pca.components_ stores top k eigen_vectors. the length of eigen_vector is the dimension of X
    #                                        #reshape to k 2-D faces
    eigen_faces = pca.components_.reshape(k, face_h, face_w)
    
    # after PCA(k).fit(X), we found k eigen_vectors(u1 matrix)
    # perform dimensionality reduction on X:
    faces_pca = pca.transform(flatten_faces)
    return faces_pca, eigen_faces

k = 150
faces_pca, eigen_faces = face_pca(faces_image, k)

# show top 12 eigen faces
fig, axes = plt.subplots(3, 4, figsize=(11, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigen_faces[i], cmap='bone')
plt.show()


# show the first 12 pictures in the result of PCA
print('dimensionality reducted whatever: ')
fig, axes = plt.subplots(3, 4, figsize=(11, 6))
for i, ax in enumerate(axes.flat):
    dr_whatever = faces_pca[i].reshape(10, -1)
    ax.imshow(dr_whatever, cmap='bone')


'''
    ------------   k_means   ---------------
'''
# store final_cluster_ids to KMeans.labels
class KMeans():
    def __init__(self, k, max_iter=500):
        self.k = k
        self.max_iter = max_iter
        # n * 1 array，store each sample's final_cluster_id
        self.labels = None
    
    def _dist(self, a, b): # a, b: 1-D vectors
        return np.math.sqrt(sum(np.power(a - b, 2)))
    
    # data: n*m array (n samples of m features each)
    # return: k random centroids
    def _init_centroids(self, data):
        m = data.shape[1]
        centroids = np.empty((self.k, m))
        for i in range(m):
            minVal = min(data[:, i])
            maxVal = max(data[:, i])
            centroids[:, i] = (minVal + (maxVal - minVal) * np.random.rand(self.k, 1)).flatten()
        return centroids
    
    '''
    perform k_means on 'data'. then save the cluster_ids into self.labels
    data: n * m.  n - sample#; m - feature#
    '''
    def fit(self, data):
        # sample count
        n = data.shape[0]
        # labels assigned for each sample
        cluster_labels = np.zeros(n)
        # n distances between each sample and the corresponding centroid
        cluster_dists = np.full(n, np.inf)
        
        # randomly initialize k centroids
        centroids = self._init_centroids(data)
        
        for _ in tqdm(range(self.max_iter)):
            # if no sample's label is changed, no more iteration needed
            cluster_changed = False
            
            # step 1. use updated centroids to assign each sample's label
            for i in range(n):
                sample = data[i, :]
                min_dist = np.inf
                min_index = -1
                # iterate all cluster centroids，try to assign this sample to a closer cluster
                for j in range(self.k):
                    centroid = centroids[j, :]
                    # distance between sample_i and centroid_j
                    dis = self._dist(centroid, sample)
                    # find the nearest centroid
                    if dis < min_dist: 
                        min_dist = dis
                        min_index = j
                
                if cluster_labels[i] != min_index and cluster_dists[i] > min_dist:
                    cluster_changed = True  # at least 1 sample's label is changed, needs more iteration
                    cluster_labels[i] = min_index 
                    cluster_dists[i] = min_dist

            if not cluster_changed:
                break
            
            # step 2. adjust all centroid by using new clusters
            for i in range(self.k):
                index = np.nonzero(cluster_labels == i)[0]
                centroids[i, :] = np.mean(data[index], axis=0)
        
        self.labels = cluster_labels        
        
# Clustering
cluster_num = 40
cluster = KMeans(k = cluster_num)
cluster.fit(faces_pca)

# Show the final results
labels = cluster.labels
for i in range(cluster_num):
    index = np.nonzero(labels==i)[0]
    num = len(index)
    this_faces = faces_image[index]
    fig, axes = plt.subplots(1, num, figsize=(4 * num, 4))
    fig.suptitle("Cluster " + str(i), fontsize=20)
    for i, ax in enumerate(axes.flat):
        ax.imshow(this_faces[i], cmap='bone')
