import numpy as np


def load_dataset(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    attributes = [list(map(float, line.strip().split(',')[:-1])) for line in lines]
    data = [line.strip().split(',') for line in lines]
    return attributes, data


def euclidean_distance(sample, centroid):
    squared_diff_sum = 0

    for attr, cent in zip(sample, centroid):
        squared_diff_sum += (float(attr) - cent) ** 2

    return squared_diff_sum ** 0.5


def calculate_distances(attributes, centroids):
    distances = []
    for centroid in centroids:
        centroid_distances = [euclidean_distance(attribute, centroid) for attribute in attributes]
        distances.append(centroid_distances)
    return distances


def assign_clusters(data, centroids):
    new_clusters = [[] for _ in range(len(centroids))]
    for sample in data:
        cid = 0
        min_distance = euclidean_distance(sample[:-1], centroids[0])
        for i in range(len(centroids)):
            distance = euclidean_distance(sample[:-1], centroids[i])
            if distance < min_distance:
                min_distance = distance
                cid = i
        new_clusters[cid].append(sample)
    return new_clusters


def initiate_clusters(data, k):
    n_samples = len(data)
    clusters = [[] for _ in range(k)]
    for attribute in data:
        cluster_id = np.random.randint(0, k)
        clusters[cluster_id].append(attribute)
    return clusters


def get_purity(clusters):
    purity = [{} for _ in range(len(clusters))]
    labels = {}
    counter = 0
    for cluster in clusters:
        for sample in cluster:
            if sample[-1] not in labels:
                labels[sample[-1]] = 0
        for label in labels:
            for sample in cluster:
                if sample[-1] == label:
                    labels[label] += 1 / len(cluster)
        dominant_label = max(labels, key=lambda k: labels[k])
        purity[counter][f'cluster {counter}'] = f'dominant_label is {dominant_label} ' \
                                                f'with {labels[dominant_label]:.2f}; {labels}'
        labels = {}
        counter += 1
    return purity


def calculate_centroids(clusters):
    centroids = []
    for cluster in clusters:
        centroid = [0] * (len(cluster[0]) - 1)

        for point in cluster:
            for i in range(len(point) - 1):
                centroid[i] += float(point[i])

        centroid = [coord / len(cluster) for coord in centroid]
        centroids.append(centroid)
    return centroids


def main():
    attributes, data = load_dataset("iris_kmeans.txt")

    k = int(input('Enter number of clusters: '))
    clusters = initiate_clusters(data, k)
    centroids = calculate_centroids(clusters)
    distances_str = calculate_distances(attributes, centroids)
    iteration = 1
    while True:
        centroids = calculate_centroids(clusters)
        distances = calculate_distances(attributes, centroids)
        sum_of_distances = np.sum(np.min(distances, axis=0))
        new_distances_str = str(distances)
        purity = get_purity(clusters)
        print(f'\nIteration: {iteration}')
        print(f'Sum of distances: {sum_of_distances:.2f}')
        for cluster in purity:
            print('Purity:', cluster)

        new_clusters = assign_clusters(data, centroids)

        if not iteration == 1 and distances_str == new_distances_str:
            break
        distances_str = new_distances_str
        clusters = new_clusters
        iteration += 1


if __name__ == "__main__":
    main()
