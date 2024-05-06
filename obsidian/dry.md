# Q1
## (a)
This classification model is not linear w.r.t. the input.
Proof (in 1d):
Let $x_1 = 1, x_2 = 2$ be arbitrary inputs, $b = 0$, $W = 1$, and $\tau = 3$. Then:
$$
\begin{align*}
&f(u + v) = \sigma(W \cdot (x_1 + x_2) + b) = \sigma(3) = 1 \\
&f(u) + f(v) = \sigma(Wx_1 + b) + \sigma(Wx_2 + b) = \sigma(1) + \sigma(2) = 0 + 0 = 0 \\
&\Rightarrow f(u + v) \neq f(u) + f(v) \Rightarrow \text{Not linear} \tag*{$\blacksquare$}
\end{align*}
$$ 
# (b)
We have 4 samples, each can be correctly or incorrectly classified, thus $\text{acc} \in \{0\%, 25\%, 50\%, 75\%, 100\%\}$. We will start by proving that $\text{acc} = 100\%$ is impossible, and we already know that every set of 3 points is shatter-able (VC dimension of 2d perceptrons is 3). Thus the maximum accuracy is $75\%$.
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
b + w_1 + w_2 > \tau \\
b + w_1 - w_2 < \tau \\
b - w_1 + w_2 < \tau \\
b - w_1 - w_2 > \tau \\
\end{cases} \\
&\implies \begin{cases}
\cancel b + \cancel w_1 - w_2 <  \cancel b + \cancel w_1 + w_2 \\
\cancel b - \cancel w_1 + w_2 <  \cancel b - \cancel w_1 - w_2
\end{cases} \\
&\implies \begin{cases}
w_2 > 0 \\
w_2 < 0
\end{cases}
\end{align*}
$$
Which is a contradiction. Therefore the maximum accuracy cannot be $100\%$, and as we mentioned above, the accuracy can be $75\%$, so the maximum accuracy is $75\%$.
$$\tag*{$\blacksquare$}$$
## (c)
thots: if this is solvable when $W_1$ brings us to 1d, then this is always solvable. Otherwise we need the answer from piazza to continue.
## (d)
In general, we'd prefer to use a linear model when:
1. The data is linearly separable.
2. 

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