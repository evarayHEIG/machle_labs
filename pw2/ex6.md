# Exercise 6

#### authors: Rafael Dousse, Eva Ray, Massimo Stefani

> a) The figure above illustrates the K-nearest neighbors (K-NN) algorithm using Euclidean distance, represented by circles. Based on a majority voting and shortest distance (when tie), which class would the “Test” instance be assigned to with K = 2, K = 5, K = 7, K = 8 ?

| K | Predicted Class | Explanation |
|---|-----------------|-------------|
| 2 | 3 | Tie between 2 and 3 but 3 is closer to the test instance. |
| 5 | 1 | There are 3 neighbors from class 1 out of 5. |
| 7 | 1 | There are 3 neighbors that belong to class 1 and 3 to class 2. There is a tie but the closest neighbor is from class 1. |
| 8 | 2 | There are 4 neighbors that belong to class 2, 3 to class 1, 1 to class 3. |

> b) Explain in your own words the differences between instance-based learning and model-based learning.

Instance-based learning stores the training examples and makes predictions using similarity to those stored instances (no global parametric model learned). The system “remembers” training points and uses them at inference.

Model-based learning develops a model (a mapping function) from training data. At test time the model (not the raw training set) is used to predict.

> c) Is K-NN an instance based learning or a model-based learning ?

K-NN is an instance based approach. It keeps the training set and classifies new points by comparing them to stored instances.

> d) When are larger K beneficial ?

A larger K can make the system more robust to noise, providing more stable predictions. It can also provides probabilistic information by using the ratio of examples for each class to indicate the confidence or ambiguity of the decision.

> e) Why are too large K detrimental ?

If K is too large, you lose locality. Indeed, neighbours far away are taken into account for the prediction. In the extreme where K tends towards N, the classifier predicts the global majority class and ignores local structure. A large K also increases computational cost.

> f) When used in classification mode, what can we do when the first categories have equal number of votes with a K-NN ?

To break ties, we can use the distance to the test point as a criterion. The class of the closest neighbor among the tied classes is chosen.

> g) Are K-NN algorithms good candidates to build a 1’000 classes image classification system ? Explain your answer.

They are generally not, because of two main reasons: 

- K-NN requires storing the entire dataset (memory heavy) and computes distances to many examples at inference (CPU heavy). This is very expensive for large datasets and many classes. 
- Pixel distance is a poor similarity metric for complex image variability. Feature extraction or learned representations are required to get good performance.

Thus, a CNN or other parametric model would be more suitable for such a large-scale classification task.

> h) Is K-NN impacted by the "curse of dimensionality" ? Explain your answer.

Yes. Distance based methods suffer from the curse of dimensionality. As dimensionality grows, the space becomes sparse and distances lose discrimination power. Additionally, covering the space adequately requires an exponentially larger number of samples. As a result, the performance of K-NN often declines because irrelevant or noisy features are included.

> i) **Difficult** What is the expected error rate computed on the training set with K = 1 ?

The expected error rate is 0%. Each training point's nearest neighbour is itself (distance 0), so the prediction is always the true label.

> j) **Difficult** What is the expected error rate computed on the training set with K = 2 and a shortest distance based tie resolution ?

The expected error rate is 0%. The two nearest neighbors of a training point are itself (distance 0) and its closest other point. There are two cases:
- If the closest other point has the same label as the training point, both neighbors vote for the correct class.
- If the closest other point has a different label, there is a tie. The tie is resolved by choosing the closest neighbor, which is the training point itself. Thus, the correct class is still predicted.