import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from utils import *

original_img=plt.imread('bird_small.png')
# Visualizing the image
plt.imshow(original_img);
# Divide by 255 so that all values are in the range 0 - 1 (not needed for PNG files)
# original_img = original_img / 255

# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

def compute_centroids(X, idx, K):
   #Returns the new centroids by computing the means of the  data points assigned to each centroid.
    # Useful variables
    m, n = X.shape
    centroids = np.zeros((K, n))
    points=[]
    for i in range(K):
        points=X[idx==i]
        
        centroids[i]=np.mean(points,axis=0)
    return centroids

def find_closest_centroids(X, centroids):
    #Computes the centroid memberships for every example
    # Set K
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distance=[]
        for j in range(centroids.shape[0]):
           norm1=np.linalg.norm(X[i]-centroids[j])
           distance.append(norm1)
                            
        idx[i]=np.argmin(distance)
    return idx

def find_closest_centroids(X, centroids):
    #Computes the centroid memberships for every example
    # Set K
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distance=[]
        for j in range(centroids.shape[0]):
           norm1=np.linalg.norm(X[i]-centroids[j])
           distance.append(norm1)
                            
        idx[i]=np.argmin(distance)
    return idx

def kMeans_init_centroids(X, K):
   # This function initializes K centroids that are to be  used in K-Means on the dataset
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids

# Run  K-Means algorithm on this data
K = 16
max_iters = 10
initial_centroids = kMeans_init_centroids(X_img, K)

# Run K-Means
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters,plot_progress=True)

print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])
# Visualize the 16 colors selected
show_centroid_colors(centroids)

# IMAGE COMPRESSION #
#idx = find_closest_centroids(X_img, centroids)

# Replace each pixel with the color of the closest centroid
X_recovered = centroids[idx, :] 

# Reshape image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape) 

# Display original image
fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
