First CNN classifier to classify between cats and dogs. Achieved 94% accuracy. 

# Neural Networks

Neural networks are computing systems with interconnected nodes that work much like neurons in the human brain. Using algorithms, they can recognize hidden patterns and correlations in raw data, cluster and classify it, and – over time – continuously learn and improve.
The original goal of the neural network approach was to create a computational system that could solve problems like a human brain. However, over time, researchers shifted their focus to using neural networks to match specific tasks, leading to deviations from a strictly biological approach. Since then, neural networks have supported diverse tasks, including computer vision, speech recognition, machine translation, social network filtering, playing board and video games, and medical diagnosis.

## There are different kinds of deep neural networks – and each has advantages and disadvantages, depending upon the use. Here, I've implemented binary classification using CNNs.
Convolutional neural networks (CNNs) contain five types of layers: input, convolution, pooling, fully connected and output. Each layer has a specific purpose, like summarizing, connecting or activating. They're very good at picking up on patterns in the input image, such as lines, gradients, circles, or even eyes and faces. It is this property that makes convolutional neural networks so powerful for computer vision. 
A convolutional neural network is a feed-forward neural network, often with up to 20 or 30 layers. The power of a convolutional neural network comes from a special kind of layer called the convolutional layer.

Convolutional neural networks contain many convolutional layers stacked on top of each other, each one capable of recognizing more sophisticated shapes. With three or four convolutional layers it is possible to recognize handwritten digits and with 25 layers it is possible to distinguish human faces.

The usage of convolutional layers in a convolutional neural network mirrors the structure of the human visual cortex, where a series of layers process an incoming image and identify progressively more complex features.

![neuron](https://64.media.tumblr.com/94ecdef944c320c962bdad9233ccadb0/tumblr_p033avqqkz1qzl9pho1_640.gifv)

# Binary classification

It is a common machine learning task. It involves predicting whether a given example is part of one class or the other. The two classes can be arbitrarily assigned either a “0” or a “1” for mathematical representation, but more commonly the object/class of interest is assigned a “1”(positive label) and the rest a “0”(negative label).

![neuron](https://64.media.tumblr.com/e3f1bbe4c2c9e5b50681c37a5522801b/408fd444f2d295d8-8e/s500x750/69156f7f224a46d8aefce07660379f0704f93439.gifv)
