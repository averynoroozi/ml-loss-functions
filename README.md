# Cross-Entropy in GPT Models

What is cross-entropy, and what makes it so central in machine learning?

Cross-entropy is a term that comes from information theory which describes a relationship between two probability distributions given a certain set of events. That is, the given set of events has probabilities assigned to each individual event in the set. 

We can think of it as follows:

[dog, cat, fish] ← A set of 3 animals - dog, cat, and fish  
We choose cat to be the correct token that we are 100% confident in.  
Say p is the distribution such that we have a 0.0 probability assigned to selecting dog, a 1.0 probability assigned to selecting cat, and a 0.0 probability assigned to selecting fish, where q is the distribution with probabilities 0.2, 0.7, 0.1, respectively.

So, in this case, p is a “one-hot” vector: [0,1,0], representing the “correct” word choice for the model. Then q is the “guesses” by the model, guessing the probabilities for each token being the correct word choice. 

In this example, the model is 70% confident in what turns out to be the right answer, which is pretty good. We should reward that, or at least give it precedence over a worse prediction. To do so, we use a loss function called cross-entropy.

We define the cross-entropy of the distribution q in relation to a distribution p as:

$$
H(p,q) = -E_p [\log q]
$$

So in the context of machine learning, the cross-entropy \(H(p,q)\) is the expected negative log probability that the model assigns to the true events. 

## Loss functions

Cross-entropy is a type of loss function, and in this case, measures how bad the model’s prediction is compared to the true answer. Intuitively, a high cost will map to a high value, where a low cost will map to a low value. How does the given function act in the way that we want, though? If we take the graph of y = -log(x) (and in this case, the constant in front of log(q) will maintain the overall function behavior, as it is always positive):



Since we only really care about how close the prediction for the correct selection is, notice that \(E_p\) will be 1 for the correct selection, but 0 for all other selections. That is, we only care about the “correct selection”, and the expression becomes

$$
H(p,q) = -\log(q)
$$

Then, probabilities closer to 1 for the correct answer get rewarded with lower output values, where values closer to 0 for the correct answer get penalized with higher output values, which is essentially how the loss is represented here. 

