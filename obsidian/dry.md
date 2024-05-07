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
b \\
w_1 \\
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
Let $W = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}, \tau = \sigma(0)$:
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

**(c)** thots: if this is solvable when $W_1$ brings us to 1d, then this is always solvable. Otherwise we need the answer from piazza to continue.

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