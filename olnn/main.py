import os
import numpy as np


class Classifier:
    def __init__(self):
        self.err = 0
        self.err_max = 10
        self.learning_rate = 0.01
        self.languages_folder = "."
        self.languages = [lang for lang in os.listdir(self.languages_folder)
                          if os.path.isdir(os.path.join(self.languages_folder, lang))]
        self.nLanguages = len(self.languages)
        self.input_size = 26
        self.output_size = self.nLanguages
        self.weights = np.random.rand(self.input_size, self.output_size)
        print(self.weights)

    def calculate_input_vector(self, text_to_process):
        input_vector = np.zeros(self.input_size)
        length = 0
        for char in text_to_process:
            val = ord(char.lower())
            if 97 <= val <= 122:
                input_vector[val - 97] += 1
                length += 1
        input_vector /= length
        return input_vector

    def train(self):
        for lang in self.languages:
            language_folder = os.path.join(self.languages_folder, lang)
            for filename in os.listdir(language_folder):
                with open(os.path.join(language_folder, filename), 'r', encoding='utf-8') as file:
                    text_to_process = file.read()
                    input_vector = self.calculate_input_vector(text_to_process)
                    mask = np.zeros(self.nLanguages)
                    mask[self.languages.index(lang)] = 1
                    net = np.dot(input_vector, self.weights)
                    desired = net * mask
                    # print("net", net)
                    output = 1 / (1 + np.exp(-net))
                    # print("output", output)
                    error_signal = (desired - output) * output * (1 - output)
                    # print("error_signal", error_signal)
                    self.weights += self.learning_rate * np.outer(input_vector, error_signal)
                    self.err += ((mask - output) ** 2) / 2
        print("error", self.err)
        # if self.err >= self.err_max:
        #     self.err = 0
        #     self.train()
        # print(self.weights)

    def classify(self, text_to_process):
        input_vector = self.calculate_input_vector(text_to_process)
        net = np.dot(input_vector, self.weights)
        print(net)
        return self.languages[np.argmax(net)]


if __name__ == "__main__":
    classifier = Classifier()
    classifier.train()

    while True:
        text = input("Enter a short text to classify (or 'quit' to quit)")
        if text.lower() == 'quit':
            break
        language = classifier.classify(text)
        while classifier.err >= classifier.err_max:
            classifier.err = 0
            classifier.train()
        print("Classified as: ", language)
