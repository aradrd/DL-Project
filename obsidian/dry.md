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
## (c)
thots: if this is solvable when $W_1$ brings us to 1d, then this is always solvable. Otherwise we need the answer from piazza to continue.
## (d)
In general, we'd prefer to use a linear model when:
1. The data is linearly separable.
2. 