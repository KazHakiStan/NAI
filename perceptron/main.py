import random
import numpy as np
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self, a, ep):
        self.ep = ep
        self.a = a
        self.weights = []
        self.threshold = 0.1
        self.sampleSpace = []
        self.testSpace = []
        self.nAttributes = None
        self.class1 = None
        self.class2 = None

    def load_training_data(self, path1, path2):
        with open(path1, 'r') as file:
            for line in file:
                sample = list(line.strip().split(','))
                self.sampleSpace.append(sample)
            self.nAttributes = len(self.sampleSpace[0]) - 1
            self.weights = [random.gauss(0.0, 0.1) for _ in range(self.nAttributes)]
            # print("Initial weights ", self.weights)
        file.close()
        self.define_classes()
        with open(path2, 'r') as file2:
            for line in file2:
                sample = list(line.strip().split(','))
                self.testSpace.append(sample)
        file2.close()

    def define_classes(self):
        if self.sampleSpace[0][-1] == "0" or self.sampleSpace[0][-1] == "1":
            self.class1 = "1"
            self.class2 = "0"
        self.class1 = self.sampleSpace[0][-1]
        for sample in self.sampleSpace:
            if sample[-1] != self.class1:
                self.class2 = sample[-1]
                break

    def train(self):
        for epoch in range(self.ep):
            for sample in self.sampleSpace:
                y = self.get_result(sample[:-1])
                diff = 1 if y == self.class2 else -1
                if y != sample[-1]:
                    for i in range(self.nAttributes):
                        self.weights[i] = self.weights[i] + diff * self.a * float(sample[i])
            if self.nAttributes == 2:
                self.display_training(epoch)
            print("Epoch number " + str(epoch + 1) + ", accuracy: " + str(self.test_accuracy()))
            random.shuffle(self.sampleSpace)
            # print("Result weights ", self.weights)

    def display_training(self, ep, *args):
        w1 = self.weights[0]
        w2 = self.weights[1]
        t = self.threshold
        x1 = np.linspace(1, 80, 1000)
        x2 = (t - x1 * w1) / w2
        plt.figure(figsize=(5, 5))
        plt.xticks(np.arange(0, 200, 20))
        plt.yticks(np.arange(0, 100, 20))
        plt.plot(x1, x2, label=f'Decision boundary epoch {ep}', color='g')
        for point in self.sampleSpace:
            if point[2] == self.class1:
                plt.scatter(float(point[0]), float(point[1]), color='b')
            else:
                plt.scatter(float(point[0]), float(point[1]), color='r')
        if len(args) != 0:
            plt.scatter(float(args[0][0]), float(args[0][1]), color='m')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Visualization of perceptron')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_result(self, sample):
        result_sum = 0
        for i in range(len(sample)):
            result_sum += self.weights[i] * float(sample[i])
        return self.class1 if result_sum >= self.threshold else self.class2

    def test_accuracy(self):
        correct_predictions = 0
        total_samples = len(self.testSpace)
        for sample in self.testSpace:
            prediction = self.get_result(sample[:-1])
            if str(prediction) == sample[-1]:
                correct_predictions += 1
        accuracy = (correct_predictions / total_samples) * 100
        return accuracy


def main():
    a = float(input("Enter the learning rate: "))
    ep = int(input("Enter number of epochs: "))
    classifier = Classifier(a, ep)
    training_file_path = input("Enter the path to the training file for class 1: ")
    test_file_path = input("Enter the path to the test file for class 1: ")
    classifier.load_training_data(training_file_path, test_file_path)
    classifier.train()

    while True:
        print("type e for exiting the program")
        option = input("Enter the observations separated by comma: ")

        if option == 'e':
            print("Exiting the program...")
            break

        observation = option.split(',')
        r = classifier.get_result(observation)
        if classifier.nAttributes == 2:
            classifier.display_training(ep, observation)
        print(f"Class {r}")


if __name__ == "__main__":
    main()
