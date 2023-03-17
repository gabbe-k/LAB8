import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 



print("lmao")

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.lr = lr
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.lru = nn.LeakyReLU()
        self.to(self.device)

    def forward(self, state):
        layer1 = self.lru(self.fc1(state))
        layer2 = self.lru(self.fc2(layer1))
        layer3 = self.dropout(layer2)
        actions = self.fc3(layer3)

        return actions
    
class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=3e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size #overwrite old memory
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # pass observation through network
            state = t.tensor(np.array([observation]), dtype=t.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            # pick maximal action
            action = t.argmax(actions).item()
        else:
            # pick random action
            action = np.random.choice(self.action_space)

        return action
    
    def learn(self):
        # only learn if we have enough memory to fill a batch
        if self.mem_cntr < self.batch_size:
            return

        # zero gradients
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        # sample random batch from memory, replace=False means no duplicates
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # create batch index
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # convert to tensors        
        state_batch = t.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = t.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = t.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = t.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        # get actions from action memory using batch
        action_batch = self.action_memory[batch]
        # estimate q value of current state + get values of actions we took
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        
        # if you are using a target network, you would use the target network here
        q_next = self.Q_eval.forward(new_state_batch)
        # set q value of next state to 0 if terminal
        q_next[terminal_batch] = 0.0

        # calculate target q value

        # q = reward + gamma * max(q_next)
        q_target = reward_batch + self.gamma*t.max(q_next, dim=1)[0]

        # calculate loss by comparing what happened to what we expected to happen (q_target vs q_eval)
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        # decrease epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


