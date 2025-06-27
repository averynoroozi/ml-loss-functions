# Cross-Entropy in GPT Models

> **Draft — Work in Progress** 
> Feedback and suggestions welcome!

## What is cross-entropy, and why is it central to machine learning?

Cross-entropy is a concept from information theory that describes a relationship between two probability distributions over the same set of events. That is, the set contains individual events, each assigned a probability.

Let's illustrate this with an example:

Suppose we have a set of three animals: **[dog, cat, fish]**. We choose **cat** to be the correct token — meaning we are 100% confident in that choice.

- Let \( p \) be the true distribution:  
  \[
  p = [0.0, 1.0, 0.0]
  \]  
  This is called a *one-hot vector*, representing the correct word choice.

- Let \( q \) be the model’s predicted probabilities:  
  \[
  q = [0.2, 0.7, 0.1]
  \]  
  Here, the model guesses there’s a 70% chance the correct token is “cat.”

Since the model is 70% confident in the correct answer, this prediction is pretty good. We want a way to reward better predictions and penalize worse ones. For this, we use a **loss function** called *cross-entropy*.

---

Mathematically, cross-entropy is defined as:

\[
H(p, q) = -\mathbb{E}_{p}[\log q] = -\sum_{x} p(x) \log q(x)
\]

In this context, \( H(p, q) \) is the expected negative log probability that the model assigns to the true events.

---

### Loss functions

Cross-entropy is a type of loss function — a function that measures how bad the model’s prediction is compared to the true answer. Intuitively, a higher loss means a worse prediction, and a lower loss means a better prediction.

To understand why the formula works, consider the graph of:

\[
y = -\log(x)
\]

where \( x \) is a probability between 0 and 1.

---

**Why this works:**  
Since \( p \) is a one-hot vector, it selects only the correct token. That means the expectation \(\mathbb{E}_p\) evaluates to 1 for the correct token and 0 for all others. So the cross-entropy simplifies to:

\[
H(p, q) = -\log(q_{\text{correct}})
\]

- When \( q_{\text{correct}} \) is close to 1 (model is confident and correct), \( -\log(q_{\text{correct}}) \) approaches 0 — meaning low loss (good).
- When \( q_{\text{correct}} \) is close to 0 (model is confident but wrong), the loss becomes very large — penalizing the model.

---

This is essentially how GPT models learn: by adjusting their parameters to minimize cross-entropy loss, they become better at predicting the correct next token.

---

*Feel free to open an issue or submit a pull request if you have suggestions or improvements!*
