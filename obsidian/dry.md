# Q1
**(a)** The sigmoid ($\sigma=\frac{1}{1+e^{-x}}$) allows us to have non-linear decision boundaries. 
This classification model is not linear w.r.t. the input, proof (in 1d):
Let $x_1 = -1, x_2 = 1$ be arbitrary inputs, $b = 0$, $W = 1$, and $\tau = 1$. Then:
$$
\begin{align*}
&f(u + v) = \sigma(W \cdot (x_1 + x_2) + b) = \sigma(0) = 0.5 \\
&f(u) + f(v) = \sigma(Wx_1 + b) + \sigma(Wx_2 + b) = \sigma(-1) + \sigma(1) = 1 \\
&\Rightarrow f(u + v) \neq f(u) + f(v) \Rightarrow \text{Not linear} \tag*{$\blacksquare$}
\end{align*}
$$ 
**(b)** We have 4 samples, each can be correctly or incorrectly classified, thus $\text{acc} \in \{0\%, 25\%, 50\%, 75\%, 100\%\}$. We will start by proving that $\text{acc} = 100\%$ is impossible, and then show a concrete example for an accuracy of $75\%$. Thus the maximum accuracy is $75\%$.
Using the bias trick (first row) we can define:
$$
\begin{align*}
&\mathbf X =
\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 1 & -1 & -1 \\
1 & -1 & 1 & -1
\end{bmatrix}
, \mathbf W = 
\begin{bmatrix}
b &
w_1 &
w_2
\end{bmatrix}
\end{align*}
$$
Assume by contradiction $\text{acc} = 100\%$:
$\sigma$ is strictly increasing and invertible thus:
$$
\begin{align*}
y =
\begin{bmatrix}
1 \\
0 \\
0 \\
1
\end{bmatrix}
 = \sigma(WX) &\implies
\begin{cases}
\sigma(b + w_1 + w_2) > \tau \\
\sigma(b + w_1 - w_2) < \tau \\
\sigma(b - w_1 + w_2) < \tau \\
\sigma(b - w_1 - w_2) > \tau \\
\end{cases} \\
&\implies \begin{cases}
\sigma(b + w_1 - w_2) < \sigma(b + w_1 + w_2) \\
\sigma(b - w_1 + w_2) < \sigma(b - w_1 - w_2) \\
\end{cases} \\
&\underbrace{\implies}_{\sigma^{-1}} \begin{cases}
\cancel b + \cancel w_1 - w_2 <  \cancel b + \cancel w_1 + w_2 \\
\cancel b - \cancel w_1 + w_2 <  \cancel b - \cancel w_1 - w_2
\end{cases} \\
&\implies \begin{cases}
w_2 > 0 \\
w_2 < 0
\end{cases}
\end{align*}
$$
$$\tag*{$\blacksquare$}$$
Which is a contradiction. Therefore the maximum accuracy cannot be $100\%$.
Let $W = \begin{bmatrix} 0 & 1 & 1 \end{bmatrix}, \tau = \sigma(0)$:
$$
y = \sigma(WX) =
\begin{bmatrix}
\sigma(2) \\
\sigma(0) \\
\sigma(0) \\
\sigma(-2)
\end{bmatrix}
\implies \begin{cases}
(1, 1) &\to \textcolor{blue}{\text{Positive}} \\
(1, -1) &\to \textcolor{blue}{\text{Positive}} \\
(-1, 1) &\to \textcolor{blue}{\text{Positive}} \\
(-1, -1) &\to \textcolor{red}{\text{Negative}}
\end{cases}
$$
This yields $75\%$ accuracy, which as we've shown is the maximum achievable accuracy.

**(c)** Yes, we can now achieve $100\%$ accuracy.
Let:
$$
\begin{align*}
&W_1 = \begin{bmatrix}
-0.99 & -1.3 \\
-0.94 & -1.0
\end{bmatrix}, W_2 = \begin{bmatrix}
-3.5 & 3.5
\end{bmatrix} \\
&b_1 = \begin{bmatrix}
2.4 & -3.2
\end{bmatrix}, b_2 = 2.7 \\
&\tau = 0
\end{align*}
$$
And so:
$$
\begin{align*}
y &= W_2 \sigma \left(W_1X^\top + b_1\right) + b_2 \\
&= \begin{bmatrix}
0.87 \\
-0.43 \\
-0.28 \\
0.005
\end{bmatrix}
\implies \begin{cases}
(1, 1) &\to \textcolor{blue}{\text{Positive}} \\
(1, -1) &\to \textcolor{red}{\text{Negative}} \\
(-1, 1) &\to \textcolor{red}{\text{Negative}} \\
(-1, -1) &\to \textcolor{blue}{\text{Positive}}
\end{cases}
\end{align*}
$$
**(d)** In general, we'd prefer to use a linear model when:
1. The data is (at least somewhat) linearly separable, consisting of distinct groups with a low amount of noise. In this case a linear decision boundary can fit the data well and achieve good results on both seen and unseen data.
2. In a case where the relation between the groups in a dataset is complex or undeterminable due to the dataset's small size, we might prefer a linear model due to its simplicity; more robust complex models might overfit the data or include strong assumptions on its distribution and fail to generalize. In this case a non-linear transformation can be used with a linear model (linear decision boundary) to improve the model's accuracy.
# Q2
The code is currently accumulating (`loss.backward()` computes and adds the gradient to the existing values) the gradients throughout different iterations of different samples. In the correct optimization process we've learned in class we've seen that we need to zero out the gradient between batches, since the previous batches' gradients may no longer point us in the optimal direction regarding the current weights and sample.
Here's a fixed code snippet:
```python
for (x, y) in dataloader:
	loss = loss_fn(model(x), y)
	optimizer.zero_grad() # This line is new!
	loss.backward()
	optimizer.step()
```

# Q3
**(a)** In the context of CNNs, the term "Receptive Field" refers to the portion of the input that affects a specific node in the network. The term originates from the context of input images where, by using convolutions, many pixels in one layer affect few pixels in the following layer, and so on.

**(b)**
- A large receptive field is useful when attempting to understand the global context of the input as a whole. e.g. when the input is an image and the network's goal is to describe the image in plain words (describe the scene).
- A small receptive field is useful when attempting to extract fine-grained features and more localized context. e.g. when trying to recognize smaller objects in a big image and not the scene as a whole (Where's Waldo?), where we want to prevent the influence of irrelevant context.

**(c)**
1. **Kernel Size:** Using a bigger kernel size allows for more pixels to affect each pixel in a subsequent layer.
2. **Dilation:** Introducing gaps between kernel elements increases the receptive field without increasing the number of inputs to each node in the following layer.
3. **Deepen the Network:** Using more convolution layers in the network allows the deeper layers to have larger receptive fields.
# Q4
**(a)** There's several problems with the approach our friend has chosen.
Firstly, computing the loss based on the discrete classification values seems to be a bad choice, since there's no way for the optimizer to incrementally improve (it only knows whether it was right or wrong). Worse yet, it seems to imply that there's something better about classifying e.g. a horse as a dog rather than classifying a horse as a cat. This is because MSE penalizes distances, but there's no meaning to the "distance" between the different classifications.

**(b)** L1 loss ignores 2 main properties of the segmentation task. The first being the same as in (a), measuring the absolute error between different labels makes no sense in the context of classification. The second being the spatial relationship between pixels; i.e. neighboring pixels often belong to the same class. Due to these reasons: penalizing all mistakes equally might not be suitable in this case (if we had prior information about the underlying data distribution it should've also been taken into account when penalizing errors).

**(c)** We'd suggest 2 changes to her method:
1. Use a feature map per class, meaning each pixel will have a probability for each classification, this will help solve the problem we mentioned in (a).
2. Based on these probabilities, use the Cross Entropy Loss function, which will help the model optimize and fit the underlying data distribution by penalizing pixels which were given a low probability for the ground truth label (and vice versa).
	- It's worth mentioning that there are other loss functions which take the spatial aspect into account. Some of them are modifications of the CE Loss function. For example, the Focal Loss, which takes into account the relationship between foreground and background.
# Q5
**(a)**
- **Vanishing Gradients** - This term refers to the phenomena where the gradients calculated using backpropagation on a neural network become increasingly small, obscuring the actual gradient step that needs to be taken. This is a result of the multiplication of small gradient values that arise from the chain rule, in combination with deep networks (the deeper the network, the more likely the problem is to occur) and the tendency of some activation functions to become relatively flat for large and small inputs.
- **Exploding Gradients** - This term refers to the phenomena where the gradients calculated using backpropagation on a neural network become increasingly large, causing the weights to change drastically between iterations of the training process and possibly circling a minima and never converging. Similarly to Vanishing Gradients, this is a result of the multiplication used in applying the chain rule while backpropagating. This can happen due to a number of reasons, namely: unfortunate weight initialization, activation function gradients, and network depth.
**(b)** 
###### Vanishing Gradients
Let's consider the following model:
$$
\hat y = \sigma_2(W_2\sigma_1(W_1x))
$$
Let $\mathcal L = L_2$, $W_1 = W_2 = x = 3$, and $y = 0$.
in the univariate case.
$$
\begin{align*}
\mathcal L &= \hat y ^ 2 \approx 0.9\\
\frac {\partial \mathcal L} {\partial W_1} &= \frac {\partial \mathcal L} {\partial \sigma_2} \cdot \frac {\partial \sigma _2 } {\partial \sigma _1} \cdot \frac {\partial \sigma _1}{\partial W_1} \\
&= 2\hat y \cdot \sigma_2\left(W_2\sigma_1\right)\cdot (1 - \sigma_2(W_2 \sigma_1)) \cdot W_2 \cdot \sigma_1\left(W_1x\right)\cdot (1-\sigma_1(W_1x))\cdot x \\
&\approx 9.5 \cdot 10^{-5}
\end{align*}
$$
As we can see, the gradient has vanished.
###### Exploding Gradients
Let's consider the following model:
$$
\hat y = \text{ReLU}_2(W_2\text{ReLU}_1(W_1x))
$$
Let $\mathcal L = L_2$, $W_1 = W_2 = 90$, $x = 1$, and $y = 0$.
in the univariate case.
$$
\begin{align*}
\mathcal L &= \hat y ^ 2 \approx 6.56 \cdot 10^7\\
\frac {\partial \mathcal L} {\partial W_1} &= \frac {\partial \mathcal L} {\partial \text{ReLU}_2} \cdot \frac {\partial \text{ReLU} _2 } {\partial \text{ReLU} _1} \cdot \frac {\partial \text{ReLU} _1}{\partial W_1} \\
&= 2\hat y \cdot \text{ReLU}_2\left(W_2\text{ReLU}_1\right)\cdot (1 - \text{ReLU}_2(W_2 \text{ReLU}_1)) \\ &\cdot W_2 \cdot \text{ReLU}_1\left(W_1x\right)\cdot (1-\text{ReLU}_1(W_1x))\cdot x \\
&\approx 1.45 \cdot 10^6
\end{align*}
$$
As we can see, the gradient has exploded.
**(c)**
- **MLP** - To avoid vanishing gradients, use a non-squashing activation function. Meaning one that doesn't have any horizontal asymptotes. As we explained in **(a)**, vanishing gradients occur as a result of activation function gradients that are close to 0, we can eliminate this problem by introducing activation functions with gradients that don't zero-out as easily.
- **CNN** - To reduce the possibility of exploding gradients, use batch normalization. This will help keep the gradients in check.
- **RNN** - 