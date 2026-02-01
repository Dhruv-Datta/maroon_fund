# LSTM Quant Model

# Overview

This model is a Recurrent Neural Network that uses quantile regression to output percentiles of stock prices. 

## Objective of Model
The output of our model consists of quantiles, percentiles representing the likelihood of reaching specific stock prices. From these quantiles, we can construct an estimated price distribution, which can then be compared to market-implied distributions derived from options data. By examining these results alongside option Greeks such as **delta** and **gamma**, we can identify potential discrepancies that may reveal opportunities the market has not yet recognized.

## Figure of Overall Model
<img width="300" height="706" alt="Screenshot 2025-08-15 at 3 19 14 PM" src="https://github.com/user-attachments/assets/a81f194b-3d58-4a20-93f0-0d47eea11cdd" />


## Quantile Regression
**Quantile regression** is a type of regression analysis that estimates conditional quantiles of a response variable, rather than just the mean. In traditional regression, the model predicts a single expected value of the output given the inputs. In contrast, quantile regression predicts specific quantiles (in our case that starts at 15th percentile and increases by 10 all the way until 85th) which represent the value below which a given percentage of the data is expected to fall.


### Derivation
What makes quantile regression especially powerful is that it does not assume any underlying distribution for the data. Unlike methods that rely on normality or other parametric assumptions, quantile regression can model outcomes even when the distribution is skewed, heavy-tailed, or otherwise irregular. 

It is based on the pinball loss function instead of traditional mean squared error which penalizes underestimates and overestimates differently based on the percentiles. 

$$
L_\tau(y, \hat{y}) =
\begin{cases} 
\tau \cdot (y - \hat{y}) & \text{if } y > \hat{y} \\
(1-\tau) \cdot (\hat{y} - y) & \text{if } y \le \hat{y}
\end{cases}
$$

Where:  
- $y$ = true value  
- $\hat{y}$ = predicted value  
- $\tau$ = desired quantile (e.g., .15, .25, .35, etc.)  

To create the quantiles, the model makes a prediction $\hat{y}$.  Based on this guess and the target quantile, the pinball loss function determines how much the model should be penalized for being above or below the true value. For example, if we want the 90th percentile, the model should prefer overestimation to underestimation. In this case, the pinball loss penalizes underestimation more heavily (gradient = 0.9) than overestimation (gradient = 0.1).

This loss is then applied to the model in the same way as MSE or cross-entropy would be: the loss value is backpropagated through the network, and the resulting gradients adjust the weights so that future predictions are nudged toward the correct quantile. By repeating this process over many samples, the predictions converge to the conditional quantile rather than the mean.

Once multiple quantiles are trained this way, we can linearly interpolate between them to construct an approximate distribution, giving us a more complete picture of the possible range of future stock prices.

### Ensuring Non-Crossing Quantiles
One challenge when predicting quantiles independently is that they may cross. For example, the model might predict the 25th percentile as 110, the 35th percentile as 95, and the 45th percentile as 115. This is illogical, since 25% of the data falling below 110 cannot be consistent with 35% of the data falling below 95.

To solve this, we apply a parametric transformation at the output layer of our model. Instead of outputting each quantile directly, the model:

- Predicts a base quantile (the lowest quantile, e.g., the 15th).

- For higher quantiles, outputs increments rather than absolute values.

- Ensures these increments are always positive by passing them through a softplus activation, and then applies a cumulative sum to construct higher quantiles.

This guarantees that each predicted quantile is greater than or equal to the previous one, eliminating crossings.

*Note* - A softplus activation function is used since it will push all values to be positive supported by the idea of delta and is used rather than ReLu because of its smooth gradients making optimization easier.

### Quantile Regression in Context
The use of quantile regression helps us understand the full distribution of our model’s predictions. If we were to output only a single point estimate, it would fail to capture the range of possible outcomes and the uncertainty inherent in the market. All prices have a probability of being reached, so by estimating percentiles, we can quantify the likelihood of different price levels. This approach allows us to construct a complete probabilistic view of future prices, providing richer insights for comparison with market data and for identifying potential opportunities that may not be apparent from a single forecast. We plan to output the 15th percentile as a base quantile, followed by seven positive deltas. Each delta is cumulatively added to the base to produce the higher quantiles, giving us predictions from the 15th to the 85th percentile in steps of 10. By outputting this, we are able to get enough understanding of our distribution without overloading our computing power. Using these 8 quantiles, we can linearly interpolate between the quantiles and create a distribution.



## Use of Neural Networks

The reason we decided to use neural networks as the basis for our model is because of the complexity of our problem. Predicting the stock market is a sophisticated problem that the world is still trying to understand. Due to the highly non-linear relationship between market variables along with the multifaceted interactions among economic, technical, and behavioral factors, traditional linear models fail to capture these patterns. Neural networks, with their ability to approximate complex nonlinear functions and learn intricate dependencies from data, are well-suited for uncovering these hidden dynamics.

### Recurrent Neural Networks:
While standard neural networks are powerful tools for modeling complex and nonlinear relationships, they have a limitation when applied to time-series data: they do not inherently account for temporal dependencies. In our case, because today’s price is strongly influenced by yesterday’s price (and prior days), we chose to use a recurrent neural network (RNN), which is specifically designed to capture sequential patterns and dependencies over time.

### Long Short-Term Memory:
<img width="2900" height="1444" alt="image" src="https://github.com/user-attachments/assets/c97f5df0-878f-4a1d-9ab0-e4a4f5b9fe9b" />

### Using an LSTM:
An LSTM (Long Short-Term Memory network) is generally better than a standard RNN because it overcomes the main weakness of RNNs: the inability to capture long-term dependencies due to the vanishing gradient problem. While a vanilla RNN quickly forgets past information as new inputs overwrite its hidden state, an LSTM introduces a cell state and gating mechanisms (input, forget, and output gates) that control what information is stored, updated, or discarded. This allows LSTMs to maintain and use relevant information across much longer sequences, making them far more effective for complex tasks like time-series forecasting, language modeling, and financial prediction where both short and long-term context matter.

### LSTM Breakdown In Context:
**States:**

Cell State (C<sub>t</sub>): The "conveyer belt" that carries long term memory. By seperating this from the output, we overcome the vanishing gradient problem. In context, this carries structural information about the time series like market cycles, risk environment, volatility clustering and more.

Hidden State (h<sub>t</sub>): This is the short-term, immediately relevant signal. It combines the long-term cell state with the current input to generate the next forecast. Even when not directly used for forecasting, the hidden state plays a key role in passing along short-term information, like recent price movements or short-term volatility, through the window to the next steps in the sequence.

**Gates:**

Forget Gate (𝒇<sub>t</sub>): This gate decides how much of the past information in the cell state should be kept or discarded. It looks at the previous hidden state and the current input to assign weights between 0 and 1 to different parts of the memory. For example, if past data showed a volatility spike but current conditions no longer reflect that regime, the forget gate can down-weight that outdated signal so the model doesn’t overemphasize it.

Input Gate (i<sub>t</sub>): This gate controls how much of the current data should be added to the long-term memory (Cell State). It evaluates the current input together with the hidden state (short-term memory) to decide what information should be written into the cell state. For example, a sudden increase in implied volatility may be stored if it signals a regime shift, but ignored if it looks like noise.

Output Gate (o<sub>t</sub>): This gate controls how much of the long-term cell state is revealed as the hidden state. It combines the current input with the memory in the cell state to decide what signal should be exposed. That hidden state is then passed forward through time (to the next step in the sequence) or, if the forecast window ends, used directly for prediction. In our model this would display as the different quantiles (15th-85th) which will then be used to create a distribution.

### Run Through of LSTM:
- Input is added to previous memory (h<sub>t-1</sub> + x<sub>t</sub>)
- Goes through Forget Gate (Activation Function = Sigmoid, so evaluates what is important)
- Input Gate receives (h<sub>t-1</sub> + x<sub>t</sub>) [Meant to update Cell State]
- Goes through input gate: Candidate state (tanh) proposes new content, sigmoid gate  scales it
- Cell State<sub>t-1</sub> + Forget Gate contribution + Input Gate contribution = Cell State (C<sub>t</sub>)
- Output Gate sent (h<sub>t-1</sub> + x<sub>t</sub>) [Meant to decide what hidden state (h<sub>t</sub>) should be]
- The output gate (sigmoid) multiplies the tanh of the updated cell state to produce the new hidden state. This decides what information from the long-term memory is exposed as the hidden state.
- The updated cell state C<sub>t</sub> is carried forward as long-term memory, while the hidden state h<sub>t</sub> is passed to the next timestep and can also be used as the model’s output
