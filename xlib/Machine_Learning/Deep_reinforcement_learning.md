# Deep Reinforcement Learning

The modeling of the problem:
Suppose a system (sometimes called evironment) can be in one state $s_t$ at time $t$, and all the possible states formed a set $S$. An agent, or machine, can interact with the system by taking action $a$ to the system, and all the possible action candidates formed a set $A$. After the agent imposed an action $a$ to the system at state $s_t$ at time $t$, the system may change to other state $s_{t+1}$ at the time $t+1$ and this transition is formulated by a probability function $\text{Pr}(s_{t+1}|s_t,a)$. To simplify the problem, we assum that the system evolution is a Markov process which means that the transition probability at the time $t+1$ is depends solely one the state at time $t$ and the action taken at that momentum, i.e. $(s_t,a)$. All the possible combination of $(s,a)$ formed a phase space $\Omega=S\otimes A$. The goal of reinforcement learning is designed to maximizing the rewards $R$ which is a function set manually to the system $R:S\to\mathbb{R}$ by extracting the transition probability from the system. 

## Q-learning

[Reference](https://en.wikipedia.org/wiki/Q-learning)

The idea of Q-learning is based on the idea of the Markov Decision Process (MDP). The evironment the agent interact with can be labeled as a sequence of $(s_i,t_i), i=1,\dots$, the state $s_i$ at time $t_i$. The action agent can make is the elements from set $A={a_i}, i=1,\dots$. A reward is a function of $R:S\to\mathbb{R}$ and the goal of agent is to maximum the rewards. So the evolution of the agent interaction with the evironment is directed by a Q-function:
$Q:S\times A\to \mathbb{R}$,
the expected rewards. Or more explicitly, $Q:S\times A\to R(S)$. The Q-function is assumed to be the probability transition function $Q(s_t,a) = \max_{\pi}\mathbb{E}[R_t|s,a, \pi]$ where $R_t$ is the rewards start from time $t$, and $\pi$ referes to the sequence of action possibly to be made by agent. Assuming the reward is linear, $R_t(s_t)=\gamma R_{t+1}(s_{t+1})+r_t$, where $r_t$ is the reward at time $t$, then

$$
R_t(s) = \sum_t^T r_t\gamma^{T-t},
$$

where $r_t = r(s_t)$ is a instant reward depends only on the current state, and $0<\gamma<1$ is a decay factor to indicate the future reward is reduced. A small $\gamma$ indicates a short term reward function (Imperically pick 0.9 to 0.99). The Q-function can be expressed iteratively and leads to the Bellman equation:

$$
Q(s_t, a) = r_t+\gamma\max_{a\in A} Q(s_{t+1},a).
$$

To train the agent, it can be done by iteratively update the Q-function (similar to the gradient decent):

$$
Q^*(s_t, a) \leftarrow Q(s_t,a)+\alpha\left\lbrace r_t+\gamma\max_{a\in A} Q(s_{t+1},a)-Q(s_t,a)\right\rbrace.
$$

where $Q^*$ stands for a new updated Q-function.

## Deep Q-network (DQN)

[Reference](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

The key point to handle the reinforcement learning is to understand the evironment by exploring the Q-function over the phase space $S\times A$, states and actions. By introducing the neural network as an approximation to the Q-function, it emable us to extend the size of the states in a scalable way. We don't need to label the states manually but let the nerual network to discover the pattern so that it may construct its own internal state set $S$. In this way, the Q-function approximated by a neural net is called a Q-net. 

Instead finding the maximum of the Q-function by using the Bellman equation, a loss function $L_i(\theta_i)$ at the $i$-th iteration is used to train the Q-net

$$
L_i(\theta_i)=\mathbb{E}_{(s,a)\in\Omega}\left\lbrace [y_i-Q(s_t,a|\theta_i)]^2 \right\rbrace,
$$

where the $\Omega= S\times A$ is the phase space coming from the direct product of the state set $S$ and action set $A$. And the $y_i$ is the updated Q value from the Bellman equatoin:

$$
y_i = \mathbb{E}\left[r(s)+\gamma\max_{a'\in A}Q(s', a'|\theta_{i-1})\Big|s,a\right],
$$

where $r(s)$ is the instance reward at state $s$. 

The gradient descendent is used to minimize the loss function:

$$
\nabla_{\theta}L_i(\theta_i) = \mathbb{E}_{(s,a\in\Omega)}\left\lbrace\Big[r+\gamma\max_{a'\in A}Q(s',a'|\theta_{i-1})-Q(s,a|\theta_i)\Big]\nabla_\theta Q(s,a|\theta_i)\right\rbrace.
$$

## Experience Replay

As the Q-net is a non-linear approach, it may introduce overfitting and may capture correlation pattern in one time sequence and causes the training unstable. For stablizing the training, decorrelation for input and outcome is crutial. Experience replay is a idea that one can store the training outcomes $e=(s_t, a, r_t, s_{t+1})$, experience, in memory so that we can randomly sample a set of $\{e_i\}, i=1,\dots$ for training the Q-net. The random sampling will decorrelate the states and rewards. 

The general idea of the DQN training algorithm can be illustrated by the following pseudo code:

```python
x = [] # raw system output
a = [] # actions
r = [] # rewards
e = () # experience ntuple

def pre(x) # a preprocess function to convert the system input to the network input, phi = pre(x);
def Qnet(phi,a, weights) # Q-net with initially randomized weights
def memory(e) # store the experience into the space D initialied.
def action(prob) # random sampling an action according to the probability distribution prob
def system(x, a) # the system emulator x2 = system(x1,a)
def reward(x)   # instance reward function
def maximum(Q) # return the maximum value of Q(x,a,weights) by looking over all possible action a.


for episode in range(M):
	raw_input[0] = x1
	pre_input[0] = pre(x1)
	for t in range(T):
		a[t] = action(prob)
		x[t+1] = system(x[t], a[t])
		r[t] = reward(x[t])
		e = (x[t], a[t], r[t], x[t+1])
		memory(e)
		batch = randomSample(D, m)
		for j in range(m):
			e = batch[j] # e = (x[j], a[j], r[j], x[j+1])
			if x[j+1] is terminated : 
				y[j] = r[j]
			else :
				y[j] = r[j]+gamma*maximum(Q(x[j+1],a, weights))
			grad = gradientDescend( (y[j]-Q(x[j],a[j],weights)^2)
			update(Q, grad)

```


## Deep neural network method

[Reference](https://kstatic.googleusercontent.com/files/2f51b2a749a284c2e2dfa13911da965f4855092a179469aedd15fbe4efe8f8cbf9c515ef83ac03a6515fa990e6f85fd827dcd477845e806f23a17845072dc7bd)

Converting a MDP problem to a supervised neural network learning makes the deep learning method could be applied directly to the reinfocement learning. To do so, we first need to setup the input and output for the neural network. The goal of the network is to archieve the output $z$ when the evolution is terminated. Hence, it can be expressed as

$$
f(s_t|\theta) = (v_t, \boldsymbol{p}),
$$

where $\theta$ is the weights in the network and $s_t$ is the state of the system at time $t$, $v_t$ is the probability to have the output $z=\pm 1$ at the termination point according to the current $s_t$, and the $p_i$ is the probability to take the action of $a_i$ to maximize the $v$. The corresponding loss function is

$$
L(\theta) = (v-z)^2 - \boldsymbol{\pi}\log\boldsymbol{p}+c\|\theta\|^2,
$$

where the first term only avaliable for the final state, the $\boldsymbol{\pi}$ is the surpervised action probability, the second term is a cross-entropy to make the network more similar to the surpervised instruction, and the last term is the $L_2$ normalization.

In practice, the instruction $\boldsymbol{\pi}$ is not easy to get. The model Alpha Zero takes the advantages that the systems they interests can be completely simulated, with afforable cost, so that the exploration can be done as much as you want. In this case, getting $\boldsymbol{\pi}$ becomes more realistic. 