import matplotlib.pyplot as plt


class Classifier:
    def __init__(self, k):
        self.k = k
        self.sampleSpace = []
        self.nAttributes = None

    def load_training_data(self, path):
        with open(path, 'r') as file:
            for line in file:
                sample = list(line.strip().split(','))
                self.sampleSpace.append(sample)
            self.nAttributes = len(self.sampleSpace[0]) - 1

    def distance(self, point1, point2):
        distance = 0.0
        for i in range(self.nAttributes):
            distance += (float(point1[i]) - float(point2[i])) ** 2
        return distance ** 1/2

    def get_neighbors(self, test_sample):
        distances = []
        for trainSample in self.sampleSpace:
            dist = self.distance(test_sample, trainSample[:-1])
            distances.append((trainSample, dist))
        distances.sort(key=lambda x: x[1])
        neighbors = [distance[0] for distance in distances[:self.k]]
        return neighbors

    def predict(self, test_sample):
        neighbors = self.get_neighbors(test_sample)
        class_votes = {}
        for neighbor in neighbors:
            decision = neighbor[-1]
            if decision in class_votes:
                class_votes[decision] += 1
            else:
                class_votes[decision] = 1
        return max(class_votes, key=class_votes.get)

    def test_accuracy(self, test_space):
        correct_predictions = 0
        total_samples = len(test_space)
        for sample in test_space:
            prediction = self.predict(sample[:-1])
            if prediction == sample[-1]:
                correct_predictions += 1
        accuracy = (correct_predictions / total_samples) * 100
        return accuracy


def main():
    k_value = int(input("Enter the value of k: "))
    classifier = Classifier(k_value)
    k_values = [k_value]
    accuracies = []
    training_file_path = input("Enter the path to the training file: ")
    classifier.load_training_data(training_file_path)

    while True:
        print("\nOptions:")
        print("a) Classification of all observations from the test set")
        print("b) Classification of the observation provided by the user")
        print("c) Change value of k")
        print("d) Exit")
        option = input("Choose an option: ").lower()

        if option == 'a':
            test_file_path = input("Enter the path to the test file: ")
            test_data = []
            with open(test_file_path, 'r') as file:
                for line in file:
                    instance = list(line.strip().split(','))
                    test_data.append(instance)
            accuracy = classifier.test_accuracy(test_data)
            accuracies.append(accuracy)
            print(f"Accuracy on test set: {accuracy:.2f}%")

        elif option == 'b':
            observation_str = input(
                f"Enter the observation with {classifier.nAttributes} features separated by comma: ")
            observation = list(map(float, observation_str.split(',')))
            prediction = classifier.predict(observation)
            print(f"Predicted label: {prediction}")

        elif option == 'c':
            k_value = int(input("Enter the new value of k: "))
            classifier.k = k_value
            k_values.append(k_value)
            print("Value of k changed.")

        elif option == 'd':
            print("Exiting the program...")
            break

        else:
            print("Invalid option. Please choose a valid option.")

        if len(k_values) == len(accuracies):
            plt.scatter(k_values, accuracies)
            plt.title('Accuracy vs. k')
            plt.xlabel('k')
            plt.ylabel('Accuracy (%)')
            plt.grid(True)
            plt.show()


if __name__ == "__main__":
    main()
