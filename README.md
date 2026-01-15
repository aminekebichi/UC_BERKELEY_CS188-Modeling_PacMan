

# UC Berkeley CS 188: Introduction to Artificial Intelligence

This repository contains our implementations of the Pac-Man projects from [UC Berkeley's CS 188: Introduction to Artificial Intelligence](https://inst.eecs.berkeley.edu/~cs188/su24/) course. These projects apply foundational AI concepts to the classic Pac-Man game, covering:

- **Search Algorithms** – DFS, BFS, UCS, A* and heuristic design
- **Adversarial Search** – Minimax, alpha-beta pruning, expectimax
- **Probabilistic Inference** – Bayes nets, exact inference, particle filtering
- **Reinforcement Learning** – Value iteration, Q-learning, function approximation
- **Machine Learning** – Perceptrons, neural networks, RNNs, CNNs

---

## Project 0: Python, Setup, & Autograder Tutorial

An introductory project to get familiar with Python and the autograder system used throughout the CS 188 course.

### Questions

| Question | File | Description |
|----------|------|-------------|
| Q1 | `addition.py` | Implement `add(a, b)` to return the sum of two numbers |
| Q2 | `buyLotsOfFruit.py` | Implement `buyLotsOfFruit(orderList)` to calculate the total cost of a fruit order |
| Q3 | `shopSmart.py` | Implement `shopSmart(orderList, fruitShops)` to find the cheapest shop for an order |

## Project 1: Search in Pacman

Build classical search agents that drive Pacman through a collection of intricate mazes. The project demonstrates how uninformed and informed search algorithms behave in grid worlds, how heuristics guide the agent, and how to reason about tradeoffs between optimality and runtime.

### Highlights

- Implement depth-first, breadth-first, uniform-cost, and A* search in `search.py`, then connect them to the `SearchAgent` to solve tiny through medium mazes.
- Design admissible heuristics to tackle the challenging corners layout and the "eat-all-the-dots" objective without expanding the entire state space.
- Compare algorithm performance using the autograder's timing and node-expansion stats to understand why heuristic guidance is essential for large mazes.

### Questions

| Question | File(s) | Description |
|----------|---------|-------------|
| Q1 | `search.py` | Code the depth-first search routine used by `SearchAgent(dfs)` |
| Q2 | `search.py` | Add breadth-first search so Pacman can find shortest unweighted paths |
| Q3 | `search.py` | Implement uniform-cost search for weighted layouts |
| Q4 | `search.py` | Implement A* search leveraging the Manhattan-distance heuristic |
| Q5 | `searchAgents.py` | Model the corners problem state space and tie it to A* |
| Q6 | `searchAgents.py` | Craft an admissible, consistent corners heuristic |
| Q7 | `searchAgents.py` | Write a heuristic that quickly eats all remaining food dots |
| Q8 | `searchAgents.py` | Explore a suboptimal search strategy that balances speed with path quality |

## Project 2: Multi-Agent Search

Design adversarial search agents for Pacman that compete against ghosts. This project explores game-theoretic decision making where Pacman must anticipate and react to opponent behavior in a dynamic, multi-agent environment.

### Highlights

- Implement minimax search to find optimal moves assuming ghosts play perfectly against Pacman.
- Add alpha-beta pruning to dramatically speed up minimax by eliminating irrelevant branches.
- Use expectimax to handle stochastic ghost behavior where outcomes are probabilistic rather than worst-case.
- Design evaluation functions that weigh food proximity, ghost danger, and capsule value to guide intelligent play.

### Questions

| Question | File | Description |
|----------|------|-------------|
| Q1 | `multiAgents.py` | Improve the `ReflexAgent` evaluation function for reactive decision-making |
| Q2 | `multiAgents.py` | Implement minimax search with multiple ghost adversaries |
| Q3 | `multiAgents.py` | Add alpha-beta pruning to minimax for efficiency |
| Q4 | `multiAgents.py` | Implement expectimax for probabilistic ghost modeling |
| Q5 | `multiAgents.py` | Design a better evaluation function for competitive play |


## Project 3: Ghostbusters

Track and hunt invisible ghosts using probabilistic inference. Pacman uses noisy distance sensors to locate ghosts he cannot see, applying Bayes Nets, exact inference, and particle filtering to pinpoint their positions.

### Highlights

- Build Bayes Net factor operations (join, eliminate, normalize) to perform variable elimination for probabilistic queries.
- Implement exact inference to maintain belief distributions over ghost locations as observations arrive and time elapses.
- Use particle filtering for approximate inference, enabling efficient tracking when exact methods become too expensive.
- Design a greedy ghost-hunting agent that moves toward the most likely ghost position.

### Questions

| Question | File(s) | Description |
|----------|---------|-------------|
| Q1 | `bayesNet.py` | Construct the Bayes Net structure for the ghost tracking problem |
| Q2 | `factorOperations.py` | Implement `joinFactors` to combine probability tables |
| Q3 | `factorOperations.py` | Implement `eliminate` to marginalize out variables |
| Q4 | `factorOperations.py` | Combine join and eliminate for variable elimination inference |
| Q5 | `inference.py` | Implement `getObservationProb` for sensor model probabilities |
| Q6 | `inference.py` | Implement `observeUpdate` for exact inference with observations |
| Q7 | `inference.py` | Implement `elapseTime` for exact inference with time steps |
| Q8 | `bustersAgents.py` | Implement `chooseAction` for the greedy ghost-hunting agent |
| Q9 | `inference.py` | Initialize particle filter with uniform distribution |
| Q10 | `inference.py` | Implement particle filter observation update with resampling |
| Q11 | `inference.py` | Implement particle filter time elapse |


## Project 4: Reinforcement Learning

Train Pacman agents with both model-based and model-free reinforcement learning so they can survive tricky grid worlds without knowing the transition model upfront.

### Highlights

- Implement a `ValueIterationAgent` to compute optimal utilities and policies for grid-world MDPs, exploring how discount, noise, and living reward interact.
- Analyze scenario tweaks such as the bridge crossing and farmer’s market layouts to reason about when optimal policies prefer risky or safe routes.
- Build a `QLearningAgent` that learns by playing Pacman, then extend it with function approximation to generalize across large state spaces.


### Questions

| Question | File(s) | Description |
|----------|---------|-------------|
| Q1 | `valueIterationAgents.py` | Implement value iteration to compute optimal policies for MDPs |
| Q2 | `analysis.py` | Tune discount, noise, and living reward to achieve specific policies on BridgeGrid and DiscountGrid |
| Q3 | `qlearningAgents.py` | Implement Q-learning update rule for model-free reinforcement learning |
| Q4 | `qlearningAgents.py` | Implement ε-greedy action selection for exploration vs exploitation |
| Q5 | `qlearningAgents.py` | Train Pacman via Q-learning to win on the smallGrid layout |
| Q6 | `qlearningAgents.py` | Implement approximate Q-learning with feature-based state representation |


## Project 5: Machine Learning

Build neural networks from scratch to solve classification and regression tasks. This project introduces the fundamentals of machine learning, from simple perceptrons to recurrent neural networks for sequence modeling.

### Highlights

- Implement a perceptron classifier and understand the basics of linear decision boundaries.
- Build feedforward neural networks using PyTorch's `Linear` layers and `relu` activations.
- Train models for non-linear regression to fit curved data distributions.
- Design a digit classification network to recognize handwritten numbers from the MNIST dataset.
- Construct a recurrent neural network (RNN) to identify the language of input words based on character sequences.

### Questions

| Question | File | Description |
|----------|------|-------------|
| Q1 | `models.py` | Implement a `PerceptronModel` for binary classification |
| Q2 | `models.py` | Build a `RegressionModel` neural network to fit non-linear data |
| Q3 | `models.py` | Create a `DigitClassificationModel` to classify handwritten digits |
| Q4 | `models.py` | Design a `LanguageIDModel` RNN to identify word languages |
| Q5 (Extra Credit) | `models.py` | Implement a `Convolve` function and `DigitConvolutionalModel` using convolutions |
