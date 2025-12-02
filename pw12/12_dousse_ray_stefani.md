# MachLe PW 12 - Report

#### authors: Rafael Dousse, Massimo Stefani, Eva Ray

## 3. Race Time Prediction

> 1. Explore different number of LSTM units, different lengths of previous data (sequence length) and training epochs. Show the configuration that performed the best. Observe the resulting complexity of the network (e.g., number of trainable parameters)

We tested different configurations of the LSTM network by varying the number of time steps, number of epochs, and number of LSTM units. Below is a summary table of our experiments along with their results:

| Time steps | Batch size | Epochs | Units | Trainable params | Training corr. | Test corr. | Observations |
|-----------|------------|--------|-------|-------------------|----------------|------------|--------------|
| 10        | 64         | 20     | 1     | 22                | 0.62           | 0.621      | MSE drops quickly and stabilizes around 5 epochs. |
| 20        | 64         | 20     | 1     | 22                | 0.313          | 0.3        | Predicted curve does not follow the ups and downs, stays almost flat. |
| 5         | 64         | 20     | 1     | 22                | 0.621          | 0.647      | MSE is still decreasing at 20 epochs, predicted curve is more reactive and follows the variations well. |
| 5         | 64         | 30     | 5     | 186               | **0.64**           | **0.65**       | MSE drops quickly, predicted curve follows the variations well. |
| 5         | 64         | 30     | 16    | 1297              | 0.638          | 0.645      | MSE drops quickly, predicted curve follows the variations well. |

The best model configuration we found was with 5 time steps, 30 epochs, and 5 LSTM units, achieving a training correlation of 0.64 and a test correlation of 0.65. That said, configurations that focused on increasing the number of units did not lead to significant improvements and neither to worse performance. 

After doing these experiments, we concluded the following:
- Increasing the number of time steps beyond 10 did not improve performance; in fact, it led to worse results. A possible explanation is that with more time steps, the model may struggle to capture relevant patterns due to increased complexity. It can maybe lead to having a view of the past that is too far away to be relevant for predicting the next speed.
- Increasing the number of epochs beyond 30 did not lead to better performance, as the MSE stabilized quickly.
- Increasing the number of LSTM units beyond 5 did not lead to significant improvements, suggesting that the model was already sufficiently complex to capture the necessary patterns in the data.
- The correlation coefficients we obtained were around 0.65, which is decent but indicates that there is still room for improvement in the model's predictive capabilities. However, the task at hand is quite challenging.

On the mse history plot below, we can see that the mse decreases quickly and stabilizes after a few (around 10) epochs. This suggests that we maybe could have stopped training earlier to save time but when we tried, it lead to slightly worse performance, probably due to the random initialization of the weights. The plot also show that there is no overfitting happening as the validation loss follows closely the training loss.

<div style="text-align:center; flex-direction: row;">
    <img src="images/history_plot.png" alt="" style="width:400"/>
</div> 

Let's take a look at the predicted speed for a random race (here, race 227) and compare it to the real speed. We can see that the predicted curve follows the real speed quite well, capturing the main variations. It is interesting to note that the predicted speed curve seems to be "delayed" compared to the real speed curve. This could be due to the fact that the model is using past data to make predictions, and there may be a lag in how quickly it can respond to changes in speed. Furthermore, the model seems to smooth out some of the more abrupt changes in speedm sometimes underestimating peaks. This could be due to the model's attempt to generalize from the training data, leading to a more averaged prediction. The absolute error plot shows that the errors are generally small, but there are some instances where the error is larger, particularly during rapid changes in speed.

<div style="text-align:center; flex-direction: row;">
    <img src="images/race_pred.png" alt="" style="width:400"/>
</div> 

To conclude, it is interesting to note that the model performs better on "easy" races, where the speed profile is relatively smooth and does not have many abrupt changes. In these cases, the model can probably capture the underlying patterns more effectively. The plot below shows the predicted speed for an easy race, an we can see that the predicted curve closely follows the real speed, with smaller errors compared to the previous example. In general, the "noisiness" of the error is related to how "noisy" the speed profile is itself.

<div style="text-align:center; flex-direction: row;">
    <img src="images/race_pred_easy.png" alt="" style="width:400"/>
</div> 

> 2. What is the largest error (speed prediction) you observed? Do you observe that most of those large errors show up for high speeds ? or low speeds? Why?

> 3. Using the predicted speeds for a given race, compute the expected time for a race and compute the difference between the real race time and the predicted race time in minutes. Provide the code of the cell that computes this prediction error.