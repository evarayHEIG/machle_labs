# MachLe PW 10 - Report

#### authors: Rafael Dousse, Massimo Stefani, Eva Ray

## 1. Digit recognition from raw data
This first part is focused on implementing three different architectures for digit recognition using the MNIST dataset. The goal is to compare the performance of different architectures and understand how the choice of layers and parameters affects the results. The model use the raw data (28x28 pixel intensities) as input features and 10 classes (digits from 0 to 9) as output.

The training set consists of 50000 samples, the validation set of 10000 samples, and the test set of 10000 samples. This is the repartition used by default in the provided notebook.

### First Model 
The first model is the architecture provided in the notebook. It is a simple multilayer perceptron. Here is the architecture:

 ```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 300)               235500    
                                                                 
 dense_1 (Dense)             (None, 10)                3010      
                                                                 
=================================================================
Total params: 238510 (931.68 KB)
Trainable params: 238510 (931.68 KB)
Non-trainable params: 0 (0.00 Byte)
``` 
With: 
- Batch size: 128
- Epochs: 5 
- Optimizer: RMSprop

We already obtain an accuracy of about 0.9801 on the training set, 0.9803 on the validation set and 0.9801 on the test set. The learning curves show, continuously decreasing training loss and validation loss also decreasing, with almost no overfitting It shows already good performance for a simple architecture. 
We can see with the confusion matrix that Errors are rare but mainly occur between digits with similar shapes:

 <div style="text-align:center; flex-direction: row;">
    <img src="figures/1_1_history_plot.png" alt="drawing" style="width:300"/>
    <img src="figures/1_1_cm.png" alt="drawing" style="width:300"/>
</div> 


### Second Model
The second model is a deeper architecture with dropout layers. Dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time, which helps prevent the model from relying too much on specific neurons. The architecture is as follows:

 ```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 300)            │       235,500 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 300)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │         3,010 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 238,510 (931.68 KB)
 Trainable params: 238,510 (931.68 KB)
 Non-trainable params: 0 (0.00 B)
 ```
We also reduce the batch size to 64 and increase the number of epochs to 50.
The training and validation curves show that the training loss decreases steadily over epochs but the validation loss starts around 0.09 and fluctuates slightly upward, stabilizing around 0.10–0.11 after epoch 10. This could indicates a small amount of overfitting. The gap remains limited, meaning the model generalizes reasonably well.

The final performance for the validation accuracy is around 0.982 and the test accuracy around 0.983. This is a small improvement compared to the first model, even though the architecture is still relatively small.
 
<div style="text-align:center; flex-direction: row;">
    <img src="figures/1_2_history_plot.png" alt="drawing" style="width:300"/>
    <img src="figures/1_2_cm.png" alt="drawing" style="width:300"/>
</div> 

![alt text](image.png)

### Third Model
The third model has a larger architecture with more neurons. We went from 300 neurons in the hidden layer to 1000 neurons and we kept the dropout layer. The architecture is as follows:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_15 (Dense)                │ (None, 1000)           │       785,000 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_9 (Dropout)             │ (None, 1000)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_16 (Dense)                │ (None, 10)             │        10,010 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total params: 795,010 (3.03 MB)
Trainable params: 795,010 (3.03 MB)
Non-trainable params: 0 (0.00 B)
```

We tried adding more layers, but it did not improve the performance, so we kept only one hidden layer with more neurons. The batch size is still 64 and the number of epochs is 30.

The train accuracy is around 0.9957, validation accuracy is around 0.9840 and the test accuracy around 0.985. As said before, we tried with more layers and other parameters, but the performance did not improve. This model has the best performance of the three models, but the improvement is quite small compared to the second model. We can also see on the learning curves that the model tends to overfit a little as the gap between training and validation loss increases over epochs. 

 <div style="text-align:center; flex-direction: row;">
    <img src="figures/1_3_history_plot.png" alt="history_plot" style="width:300"/>
    <img src="figures/1_3_cm.png" alt="cm" style="width:300"/>
</div> 

### Questions

> a. Select a neural network topology and describe the inputs, indicate how many are they, and how many outputs.
 
The input of all three models consists of 784 features, corresponding to the 28×28 pixel intensity values of each MNIST image. The output layer always contains 10 neurons, one per digit class (0–9), using a `softmax` activation function.
Each model differs only in the number of hidden units and dropout layers:
- Model 1: 784 -> 300 -> 10
- Model 2: 784 -> 300 -> Dropout(0.5) -> 10
- Model 3: 784 -> 1000 -> Dropout(0.5) -> 10
  
> b. Compute the number of weights of each model (e.g., how many weights between the input and the hidden layer, how many weights between each pair of layers, biases, etc..) and explain how you get to the total number of weights.

We compute weights as: $weights = (inputs + 1) × neurons$ where the "+1" corresponds to the bias term for each neuron.
**Model 1** 
*Input -> Hidden (784 -> 300)*
$(784+1)×300=785×300=235 500$
*Hidden -> Output (300 -> 10)*
$(300+1)×10=301×10=3 010$
$Total\ weights = 238 510$

**Model 2**:

Dropout does not add parameters.
So Model 2 has the exact same number of weights as Model 1:
$Total\ weights = 238 510$

**Model 3**: 

*Input -> Hidden (784 -> 1000)*
$(784+1)×1000=785000$
*Hidden -> Output (1000 -> 10)*
$(1000+1)×10=1001×10=10010$
$Total\ weights = 795 010$


> c. Comment the differences in results for the three models. Are there particular digits that are frequently  confused?

|Model|	Validation Accuracy|	Test Accuracy|	Remarks|
|-----|---------------------|--------------|--------|
|1    |	~0.980              |	~0.980     | Good baseline, very limited overfitting.|
|2    |	~0.982              |	~0.983     | Slight improvement thanks to dropout + longer training.|
|3    |	~0.984              |	~0.985     | Best accuracy, but only marginally better. Overfitting starts to appear.|
Increasing the number of neurons improves performance, but the gains become very small because MNIST is a relatively simple dataset and a shallow MLP already performs very well.

Across all three confusion matrices, the same patterns appear:
`3-5`, `5-3`, `4-9`, `9-4`, `3-9` and `2-7` are the most confused digit pairs.
The pairs might be confused due to similar curvature and orientation or share similar straight lines. For example, some handwritten 3 can look like a 5. For the 4 and the 9 pair, the top part of the 4 can be mistaken for the loop of the 9. 

Model 3 reduces the number of mistakes but does not eliminate these specific ambiguous cases.

## 2. Digit recognition from features of the input data

### First Model 

### Second Model

### Third Model

### Questions

> a. Select a neural network topology and describe the inputs, indicate how many are they, and how many outputs.

> b. Compute the number of weights of each model (e.g., how many weights between the input and the hidden layer, how many weights between each pair of layers, biases, etc..) and explain how you get to the total number of weights.

> c. Comment the differences in results for the three models. Are there particular digits that are frequently  confused?

## 3. Convolutional neural network digit recognition

In this section, we implement three different CNN architectures for digit recognition using the MNIST dataset. The goal is to compare the performance of different architectures and understand how the choice of layers and parameters affects the results. All the following models use pixels from 28x28 images as input data, which represents an input size of 784 features. The possible outputs are the digits from 0 to 9, resulting in 10 output classes.

The training set consists of 50000 samples, the validation set of 10000 samples, and the test set of 10000 samples. This is the repartition used by default in the provided notebook.

### First Model 

The first analyzed model is the model given by default in the notebook. It has the following architecture:

 ```
 _________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 l0 (InputLayer)             [(None, 28, 28, 1)]       0         
                                                                 
 l1 (Conv2D)                 (None, 28, 28, 9)         234       
                                                                 
 l1_mp (MaxPooling2D)        (None, 14, 14, 9)         0         
                                                                 
 l2 (Conv2D)                 (None, 14, 14, 9)         2034      
                                                                 
 l2_mp (MaxPooling2D)        (None, 7, 7, 9)           0         
                                                                 
 l3 (Conv2D)                 (None, 7, 7, 16)          1312      
                                                                 
 l3_mp (MaxPooling2D)        (None, 3, 3, 16)          0         
                                                                 
 flat (Flatten)              (None, 144)               0         
                                                                 
 l4 (Dense)                  (None, 25)                3625      
                                                                 
 l5 (Dense)                  (None, 10)                260       
                                                                 
=================================================================
Total params: 7465 (29.16 KB)
Trainable params: 7465 (29.16 KB)
Non-trainable params: 0 (0.00 Byte)
 ```

 - Batch size: 128
 - Epochs: 5 
 - Optimizer: RMSprop

 The accuracy of this model on the train set is about 0.9742, which is a strong start. As we can see on the history plot below, the model is not overfitting and the validation loss is still decreasing at the end of the training, so we can try to increase the number of epochs to improve the performance.

 <div style="text-align:center; flex-direction: row;">
    <img src="figures/3_1_history_plot.png" alt="drawing" style="width:300"/>
    <img src="figures/3_1_cm.png" alt="drawing" style="width:300"/>
</div> 

### Second Model

For the second model, we first simply increased the epochs to 50. However, the model tends to overfit pretty quickly, so we added Dropout layers after the flatten layer and after the first dense layer with a dropout rate of 0.3. This helps with overfitting because it randomly sets a fraction of input units to 0 at each update during training time, which helps prevent the model from relying too much on specific neurons. With this new architecture, we obtained an accuracy of 0.9862 on the test set, which is a significant improvement considering the high accuracy of the first model. The history plot and confusion matrix are shown below:

<div style="text-align:center; flex-direction: row;">
    <img src="figures/3_2_history_plot.png" alt="drawing" style="width:300"/>
    <img src="figures/3_2_cm.png" alt="drawing" style="width:300"/>
</div> 

### Third Model

For the third model, the goal is to try a more drastic change in the architecture like adding more filters, changing the kernel size, adding more convolutional layers, changing the batch size, or adding batch normalization layers. In the table below, we summarize the different architectures we tried as well as their performance on the test set and our observations regarding overfitting. 

| Test Accuracy | Architecture                                                                                                 | Observations                       |
|---------------|------------------------------------------------------------------------------------------------------------------------|------------------------------------|
| 0.9944        | 32 filters, 5x5 → 64 filters, 5x5 → 128 filters, 3x3, batch size 128                                                   | Overfitting quickly                |
| 0.9922        | 32 filters, 3x3 → 64 filters, 3x3 → 128 filters, 3x3, batch size 128                                                   | Overfitting still present but less |
| 0.9936        | 32 filters, 5x5 → 64 filters, 5x5 → 128 filters, 3x3, batch size 256                                                   | Almost no overfitting              |
| **0.9938**        | 32 filters, 3x3 → 64 filters, 3x3 → 128 filters, 3x3, batch size 256, SGD optimizer (momentum 0.9)                                 | **Almost no overfitting**              |
| 0.9923        | 32 filters, 3x3 → 64 filters, 3x3 → 128 filters, 3x3, batch normalization after each convolution layer, batch size 128 | Overfitting still present but less |
| 0.9915        | 32 filters, 3x3 → 64 filters, 3x3 → 128 filters, 3x3 → 256 filters, 3x3, batch size 128                                | Overfitting very quickly           |

The best performing model is the one with 32 filters of size 3x3 in the first convolutional layer, followed by 64 filters of size 3x3 in the second convolutional layer, and finally 128 filters of size 3x3 in the third convolutional layer. We also used a batch size of 256, the SGD optimizer with a momentum of 0.9, dropout layers with a dropout rate of 0.3 after the flatten layer and after the first dense layer and 50 epochs. This model achieved an accuracy of 0.9938 on the test set, which is a significant improvement over the previous models. The test score is 0.0212. The history plot and confusion matrix are shown below:

<div style="text-align:center; flex-direction: row;">
    <img src="figures/3_33_history_plot.png" alt="drawing" style="width:300"/>
    <img src="figures/3_33_cm.png" alt="drawing" style="width:300"/>
</div> 

### Questions

> a. Select a neural network topology and describe the inputs, indicate how many are they, and how many outputs.

The selected neural network topology is the third model described above.

- **Inputs**: The input to the model is a grayscale image of size 28x28 pixels. Each pixel is represented by a single value (intensity) ranging from 0 to 255. Therefore, the total number of input features is 28 * 28 = 784.
- **Outputs**: The output of the model is a probability distribution over 10 classes, corresponding to the digits 0 to 9. Therefore, there are 10 output neurons in the final dense layer, each representing the probability of the input image belonging to one of the digit classes.

> b. Compute the number of weights of each model (e.g., how many weights between the input and the hidden layer, how many weights between each pair of layers, biases, etc..) and explain how you get to the total number of weights.

We will go through the calculation of the number of weights for each layer in the selected model:
- **First Convolutional Layer (Conv2D with 32 filters of size 3x3)**: (3 * 3 * 1) * 32 + 32 = 320
- **Second Convolutional Layer (Conv2D with 64 filters of size 3x3)**: (3 * 3 * 32) * 64 + 64 = 18,496
- **Third Convolutional Layer (Conv2D with 128 filters of size 3x3)**: (3 * 3 * 64) * 128 + 128 = 73,856
- **First Dense Layer (Dense with 25 neurons)**: (3 * 3 * 128) * 25 + 25 = 28,825
- **Second Dense Layer (Dense with 10 neurons)**: (25) * 10 + 10 = 260
- **Total number of parameters in the model**: 320 + 18,496 + 73,856 + 28,825 + 260 = 121,757

Basically, we have:
- For convolutional layers: (filter_height * filter_width * input_channels) * number_of_filters + number_of_filters (for biases)
- For dense layers: (number_of_input_neurons) * (number_of_output_neurons) + number_of_output_neurons (for biases)

> c. Comment the differences in results for the three models. Are there particular digits that are frequently  confused?

The first model achieved a good accuracy of 0.9742, which indicates that CNNs are effective for digit recognition tasks. However, there was still room for improvement. The second model, which incorporated dropout layers to mitigate overfitting, significantly improved the accuracy to 0.9862, showing that the model benefited from being trained for more epochs and that dropout helped avoid overfitting. Finally, the third model, which involved more extensive architectural changes such as increasing the number of filters, changing kernel sizes, and using a different optimizer (SGD with momentum), achieved the highest accuracy of 0.9938. This demonstrates that deeper architectures with more parameters can capture more complex features in the data, leading to better performance, although it can also lead to overfitting. In our case, we can see on the loss history plot that the model almost don't overfit at all, which is due to the dropout layers and the larger batch size. This is ideal, since it means the model generalizes well to unseen data.

<div style="text-align:center; flex-direction: row;">
    <img src="figures/3_33_history_plot.png" alt="drawing" style="width:300"/>
    <img src="figures/3_33_cm.png" alt="drawing" style="width:300"/>
</div> 

Now, let's take a look at the confusion matrix of the third model. We can see that there are very few misclassifications but when they do occur, they tend to happen between digits that are visually similar. For example, '5' and '3', '2' and '7',  '4' and '9', '5' and '6' and '9' and '5' are sometimes confused with each other. This is likely due to the fact that these digits share similar shapes and features, making it challenging for the model to distinguish between them in certain cases. These misclassifications indicates that the model could benefit from further fine-tuning or additional training data to better learn the features that differentiate these similar digits. However, with an accuracy of 0.9938, it will be difficult to achieve significant improvements, as the model is already performing at a very high level. Furthermore, overfitting is a real challenge for this task when trying to push the accuracy even higher, as it is quite hard to keep it under control.

## 4. Chest X-ray to detect pneumonia

> Train a CNN for the chest x-ray pneumonia recognition. In order to do so, complete the code to reproduce the architecture plotted in the notebook. Present the confusion matrix, accuracy and F1-score of the validation and test datasets and discuss your results.

## General questions

### Question 1

> What is the learning algorithm being used to train the neural networks?

The algorithm used by default to optimize the weights is RMSprop. RMSprop is an optimization algorithm used to adjust the weights of a neural network. It adapts the learning rates of the weights by using a moving average of the squared previous gradients, which allows faster convergence and improves learning performance.

<div style="text-align:center">
    <img src="figures/rmsprop_equation.png" alt="drawing" style="width:300"/>
</div>

where:

- $E[g]$ is the moving average of the squared gradients
- $\frac{\delta c}{\delta w}$ is the gradient of the cost function with respect to the weight
- $\eta$ is the learning rate
- $\beta$ is the moving average parameter

> What are the parameters (arguments) being used by that algorithm?

The parameters used in the ``RMSprop`` optimizer can be found in the `Keras` documentation and are as follows:

```python
keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    loss_scale_factor=None,
    gradient_accumulation_steps=None,
    name="rmsprop",
    **kwargs
)
```

> What cost function is being used? 

The cost function used is the categorical cross-entropy loss function. This function is commonly used for multi-class classification problems, where the goal is to predict the class label of an input from multiple possible classes. The categorical cross-entropy loss measures the difference between the predicted probability distribution and the true distribution (one-hot encoded labels) and penalizes incorrect predictions more heavily. The equation for the categorical cross-entropy loss is given by:

$$L = -\sum(y \cdot \log(\hat{y}))$$

where:
- $L$ is the loss
- $y$ is the true label (one-hot encoded)
- $\hat{y}$ is the predicted probability distribution

> How did you create the training, validation and test datasets.

### Question 2

See in each experiment section.

### Question 3

> Do the deep neural networks have much more “capacity” (i.e., do they have more weights?) than the shallow ones? Explain with one example.

In general, deeper convolutional neural networks tend to have more weights than shallow models. This comes from the fact that deeper CNNs usually include a larger number of layers, and each convolution layer contains several filters that hold weights. However, this is not always true and depends on the specific model configuration. For example, a shallow model with several layers and many neurons may end up with more weights than a CNN that has only one convolution layer using small filters. 

### Question 4

See the section for part 4.

