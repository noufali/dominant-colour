"""
Kmeans clustering algorithm for colour detection in images

Initialise a kmeans object and then use the run() method.
Several debugging methods are available which can help to
show you the results of the algorithm.
"""

from PIL import Image
import random
import numpy

#creating a cluster of similar pixels - unsupervised learning because data does not have prescribed labels nor class values
class Cluster(object):

    def __init__(self):
        self.pixels = []
        self.centroid = None

    def addPoint(self, pixel):
        self.pixels.append(pixel)

    def setNewCentroid(self):

        R = [colour[0] for colour in self.pixels]
        G = [colour[1] for colour in self.pixels]
        B = [colour[2] for colour in self.pixels]

        R = sum(R) / len(R)
        G = sum(G) / len(G)
        B = sum(B) / len(B)

        self.centroid = (R, G, B)
        self.pixels = []

        return self.centroid

#running a kmeans algorithm on the clusters in order to assign centroids and then move centroids to the average center of each cluster
class Kmeans(object):
    #initialize 3 cluster groups with a size of 200 pixels
    def __init__(self, k=3, max_iterations=5, min_distance=5.0, size=200):
        self.k = k
        self.max_iterations = max_iterations
        self.min_distance = min_distance
        self.size = (size, size)

    def run(self, image):
        self.image = image
        #creating a 200x200 thumbnail image
        self.image.thumbnail(self.size)
        #creates an array of pixel values with a data type of unsigned integers (0 to 255)
        self.pixels = numpy.array(image.getdata(), dtype=numpy.uint8)
        #creating an empty list of only three cluster holders
        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        #returns a k (3) length list of unique elements chosen from the pixels array.
        randomPixels = random.sample(self.pixels, self.k)

        #assigning center of cluster to random 3 pixels we just chose
        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]

        iterations = 0

        #while iterations is less or equal to 5 continue checking distances
        while self.shouldExit(iterations) is False:
            #put cluster centroids in an array called old clusters
            self.oldClusters = [cluster.centroid for cluster in self.clusters]

            print iterations
            #assign pixels a clusters based on their distance
            for pixel in self.pixels:
                self.assignClusters(pixel)

            for cluster in self.clusters:
                cluster.setNewCentroid()

            iterations += 1

        return [cluster.centroid for cluster in self.clusters]

    def assignClusters(self, pixel):
        #infinite float value - This is useful for finding lowest values for something. for example, calculating path route costs when traversing treesself.
        #Ex: Finding the "cheapest" path in a list of options
        shortest = float('Inf')
        for cluster in self.clusters:
            distance = self.calcDistance(cluster.centroid, pixel)
            if distance < shortest:
                shortest = distance
                nearest = cluster
        #add that pixel to the nearest cluster
        nearest.addPoint(pixel)

    #calculating distance hypotenuse
    def calcDistance(self, a, b):

        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    def shouldExit(self, iterations):

        if self.oldClusters is None:
            return False

        #calculate the distance between new cluster center and old cluster center
        for idx in range(self.k):
            dist = self.calcDistance(
                numpy.array(self.clusters[idx].centroid),
                numpy.array(self.oldClusters[idx])
            )
            if dist < self.min_distance:
                return True

        if iterations <= self.max_iterations:
            return False

        return True

    # ############################################
    # The remaining methods are used for debugging
    def showImage(self):
        self.image.show()

    def showCentroidColours(self):

        for cluster in self.clusters:
            image = Image.new("RGB", (200, 200), cluster.centroid)
            image.show()

    def showClustering(self):

        localPixels = [None] * len(self.image.getdata())

        for idx, pixel in enumerate(self.pixels):
                shortest = float('Inf')
                for cluster in self.clusters:
                    distance = self.calcDistance(cluster.centroid, pixel)
                    if distance < shortest:
                        shortest = distance
                        nearest = cluster

                localPixels[idx] = nearest.centroid

        w, h = self.image.size
        localPixels = numpy.asarray(localPixels)\
            .astype('uint8')\
            .reshape((h, w, 3))

        colourMap = Image.fromarray(localPixels)
        colourMap.show()


def main():

    image = Image.open("images/workday_web.jpg")

    k = Kmeans()

    result = k.run(image)
    print result

    k.showImage()
    k.showCentroidColours()
    k.showClustering()

if __name__ == "__main__":
    main()
