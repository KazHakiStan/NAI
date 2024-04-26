import numpy as np


def load_dataset(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    attributes = [list(map(float, line.strip().split(',')[:-1])) for line in lines]
    data = [line.strip().split(',') for line in lines]
    return attributes, data


def euclidean_distance(attribute, centroid):
    squared_diff_sum = 0

    for attr, cent in zip(attribute, centroid):
        squared_diff_sum += (attr - cent) ** 2

    return squared_diff_sum ** 0.5


def calculate_distances(attributes, centroids):
    distances = []
    for centroid in centroids:
        centroid_distances = [euclidean_distance(attribute, centroid) for attribute in attributes]
        distances.append(centroid_distances)
    return distances


def assign_clusters(attributes, centroids):
    # Convert attributes to list of lists of numbers
    attributes = [[float(val) for val in sublist] for sublist in attributes]

    # Convert centroids to list of lists of numbers
    centroids = [[float(val) for val in sublist] for sublist in centroids]
    assigned_clusters = []
    for attribute in attributes:
        min_distance = float('inf')
        min_cluster_idx = None

        for idx, centroid in enumerate(centroids):
            distance = euclidean_distance(attribute, centroid)
            if distance < min_distance:
                min_distance = distance
                min_cluster_idx = idx

        assigned_clusters.append(min_cluster_idx)

    return assigned_clusters


def initiate_clusters(attributes, k):
    n_samples = len(attributes)
    clusters = [[] for _ in range(k)]
    for attribute in attributes:
        cluster_id = np.random.randint(0, k)
        clusters[cluster_id].append(attribute)
    return clusters


# def get_purity(labels, clusters):
#     purity = {}
#     n_samples = len(labels)
#     # labels = np.array(labels)
#     for i, cluster in enumerate(clusters):
#         cluster_labels = []
#
#         unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
#         dominant_label = unique_labels[np.argmax(label_counts)]
#         percentage = label_counts.max() / len(cluster_labels) * 100
#         purity[f'cluster{i + 1}'] = f'{percentage:.2f}% {dominant_label}'
#     return purity


def calculate_centroids(clusters):
    centroids = []
    for cluster in clusters:
        centroid = [0] * len(cluster[0])

        for point in cluster:
            for i, attribute in enumerate(point):
                centroid[i] += attribute

        centroid = [coord / len(cluster) for coord in centroid]
        centroids.append(centroid)
    return centroids


def main():
    attributes, data = load_dataset("iris_kmeans.txt")

    print(attributes)
    print(data)

    k = int(input('Enter number of clusters: '))
    clusters = initiate_clusters(attributes, k)
    print(clusters)
    print('Initial centroids:', calculate_centroids(clusters))
    iteration = 1
    while True:
        centroids = calculate_centroids(clusters)
        distances = calculate_distances(attributes, centroids)
        print(distances)
        sum_of_distances = np.sum(np.min(distances, axis=0))
        # purity = get_purity(labels, clusters)
        print(f'\nIteration: {iteration}')
        print(f'Sum of distances: {sum_of_distances:.2f}')
        # print('Purity:')
        # for cluster, percentage in purity.items():
        #     print(f'{cluster}: {percentage}')

        new_clusters = assign_clusters(attributes, centroids)
        if np.array_equal(clusters, new_clusters):
            break
        clusters = new_clusters


if __name__ == "__main__":
    main()
