## Transfer-learning-using-the-VGG16-model

## Project Description:
* This project implements transfer learning using the VGG16 model to classify images of cats and dogs. 
* Instead of training a convolutional neural network from scratch, I leverage the pre-trained VGG16 model, which has been trained on ImageNet, and fine-tune it for our * specific dataset.

## Key Features:
* Utilizes VGG16 as a feature extractor, with pre-trained weights from ImageNet.
* Used ImageDataGenerator to produce batches of enhanced images
* Preprocesses images (resizing, normalization)
* Fine-tunes the fully connected layers for binary classification (cat vs. dog)
* Uses Keras and TensorFlow for model implementation.
* Uses ReLU activation for hidden layers and softmax for classification
* Trains using Adam stochastic gradient descent, categorical cross-entropy loss
* loads the trained model for future predictions
* Classifies the image given as dog or cat

## Technologies Used:
* Backend: Python
* Data Handling: Pandas ,Numpy for model building
* Tools: Jupyter Notebook
* APPS: VGG16 model 
* Neural Network: CNN for Image classification
