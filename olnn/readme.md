# Implement a 1-layer neural network recognizing the language of a given text .
Neural network has as many neurons as the number of unique languages in a dataset. Neural network should recognize this number automatically based on the training data.
## Input vectors:
Input vector represents the proportion of each ascii letter in a given text.
## example:
input text: “tools of ai"
#### vector 1/9 0 0 0 0 1/9 0 0 1/9 0 0 1/9 0 0 1/3 0 0 0 1/9 1/9 0 0 0 0 0 0
## Output:
Each neuron calculates its linear output (net). Use the maximum selector to find which neuron is activated (1) and assume that other neurons are not activated (0).
## Training:
Delta rule can be used.After each training step weight vectors may be normalized to improve classification.
## Data:
Create a training dataset by yourself. Create 3-4 separate folders and name them to represent languages (pl, en, de ...) . Each language folder contains text files (10+) in a specified language - one file contains a few paragraphs of text.
Make sure that the language you choose can be represented by ascii.
## Testing:
Provide an interface to input a short text that will be classified by your program. 