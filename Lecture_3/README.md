# Lecture 3 - Linear Classifiers

Neural Networks built upon linear classifiers

CIFAR 10 dataset is used


## Parametric Approach

images as inputs
array of 32x32x3 numbers 
(3072 numbers total)
    | 
    f(x,W) 
W parameters of weights
    |
outputs 10 numbers giving class scores

            (3072,)
f(x,W) = Wx        +    b (10,)
(10,)   (10,3072)

Linear Classifier :Predictions are Linear
    f(x,W) = Wx  (ignore bias)

    f(cx,W) = W(cx) = c*f(x,W)

    linearity property 