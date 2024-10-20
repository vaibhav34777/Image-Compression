## Image Compression using K-Means Clustering ##
### Overview ###
This project demonstrates image compression using the K-Means clustering algorithm. By representing an image with a reduced number of colors, we can compress the image while retaining its essential visual features. This method involves clustering the pixel colors in an image and replacing them with the colors of the centroids of the clusters.
### Dataset ###
Any standard image in formats like JPEG, PNG, or BMP can be used for this project. The input image will be read, and the K-Means algorithm is applie
### Approach ###
Image Representation: The image is represented as a 2D array of pixel values, where each pixel has RGB values.
K-Means Clustering:
Select K as the number of clusters (which determines the number of colors in the compressed image).
Apply K-Means to cluster the pixel values based on their RGB similarities.
Each cluster centroid represents a color that will replace all pixel values in that cluster.
Image Reconstruction: Using the K cluster centroids, reconstruct the image with reduced colors.
Compression Comparison: Compare the original image with the compressed one in terms of visual quality and size.
### Results ###
The project demonstrates a significant reduction in the number of colors used in the image while maintaining visual similarity to the original.
As the value of K increases, the compressed image retains more of the original quality but results in less compression.
