# Important equations for backpropagation

###### General definitions

$$
\delta^L = error \quad
C = (a^L - y)^2 \quad
\nabla_a C = 2(a^L - y) \\
a^L = \sigma(a^{L-1} w^L + b^L) = \sigma(z^L) \quad z^L = a^{L-1} w^L + b^L
$$

###### Defining and deriving error

$$
\delta^L_j = \frac{\part C}{\part z^L_j} = \frac{\part C}{\part a^L_j} \frac{\part a^L_j}{\part z^L_j} \\
\delta^L = \frac{\part C}{\part a^L} \frac{\part a^L}{\part z^L} \quad
\frac{\part C}{\part a^L} = \nabla_a C \quad
\frac{\part a^L}{\part z^L} = \sigma\prime(z^L)\\
\delta^L = \nabla_a C \odot \sigma\prime(z^L) \\
\delta^L = 2(a^L - y)\odot \sigma\prime(z^L)
$$

###### Error of the previous layer based on attributes of the current layer

$$
\delta^{L-1} = \frac{\part C}{\part z^{L-1}} = \frac{\part C}{\part z^L} \frac{\part z^L}{\part z^{L-1}} \\
\frac{\part C}{\part z^L_j} = \delta^L \quad
z^L = w^L \sigma(z^{L-1}) + b^L \quad 
\frac{\part z^L}{\part z^{L-1}} = w^L \sigma\prime(z^{L-1})\\
\delta^{L-1} = \delta^L w^L \sigma\prime(z^{L-1})\\
\delta^{L-1} =((w^L)^T\space\delta^L) \odot\sigma\prime(z^{L-1})
$$

###### How weights affect the cost function

$$
\frac{\part C}{\part w^L} = \frac{\part C}{\part z^L} \frac{\part z^L}{\part w^L} \quad
\frac{\part C}{\part z^L} = \delta^L \quad
\frac{\part z^L}{\part w^L} = a^{L-1} \\
\frac{\part C}{\part w^L} = a^{L-1} \delta^L\\
\frac{\part C}{\part w^L_{jk}} = a^{L-1}_k \space \delta^L_j \quad
\frac{\part C}{\part w^L} = a^{L-1} \space \delta^L \quad
\frac{\part C}{\part w} = a_{in} \space \delta_{out}
$$

###### How biases affect the cost function

$$
\frac{\part C}{\part b^L} = \frac{\part C}{\part z^L} \frac{\part z^L}{\part b^L} \quad
\frac{\part C}{\part z^L} = \delta^L \quad
\frac{\part z^L}{\part b^L} = 1 \\
\frac{\part C}{\part b^L} = \delta^L
$$

###### Gradient descent

$$
w^L = w^L - \eta \frac{\part C}{\part w^L} = w^L - \eta a^{L-1} \delta^L \\
b^L=b^L - \eta \frac{\part C}{\part b^L} =b^L - \eta \delta^L
$$