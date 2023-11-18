#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import random
import matplotlib.pyplot as plt # Graphical library
from sklearn.metrics import mean_squared_error # Mean-squared error function


# # Coursework 1 :
# See pdf for instructions. 

# In[2]:


# WARNING: fill in these two functions that will be used by the auto-marking script
# [Action required]

def get_CID():
  return "02255048" # Return your CID (add 0 at the beginning to ensure it is 8 digits long)

def get_login():
  return "cl222" # Return your short imperial login


# ## Helper class

# In[3]:


# This class is used ONLY for graphics
# YOU DO NOT NEED to understand it to work on this coursework

class GraphicsMaze(object):

  def __init__(self, shape, locations, default_reward, obstacle_locs, absorbing_locs, absorbing_rewards, absorbing):

    self.shape = shape
    self.locations = locations
    self.absorbing = absorbing

    # Walls
    self.walls = np.zeros(self.shape)
    for ob in obstacle_locs:
      self.walls[ob] = 20

    # Rewards
    self.rewarders = np.ones(self.shape) * default_reward
    for i, rew in enumerate(absorbing_locs):
      self.rewarders[rew] = 10 if absorbing_rewards[i] > 0 else -10

    # Print the map to show it
    self.paint_maps()

  def paint_maps(self):
    """
    Print the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders)
    plt.show()

  def paint_state(self, state):
    """
    Print one state on the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    states = np.zeros(self.shape)
    states[state] = 30
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders + states)
    plt.show()

  def draw_deterministic_policy(self, Policy):
    """
    Draw a deterministic policy
    input: Policy {np.array} -- policy to draw (should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    for state, action in enumerate(Policy):
      if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
        continue
      arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
      action_arrow = arrows[action] # Take the corresponding action
      location = self.locations[state] # Compute its location on graph
      plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
    plt.show()

  def draw_policy(self, Policy):
    """
    Draw a policy (draw an arrow in the most probable direction)
    input: Policy {np.array} -- policy to draw as probability
    output: /
    """
    deterministic_policy = np.array([np.argmax(Policy[row,:]) for row in range(Policy.shape[0])])
    self.draw_deterministic_policy(deterministic_policy)

  def draw_value(self, Value):
    """
    Draw a policy value
    input: Value {np.array} -- policy values to draw
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    for state, value in enumerate(Value):
      if(self.absorbing[0, state]): # If it is an absorbing state, don't plot any value
        continue
      location = self.locations[state] # Compute the value location on graph
      plt.text(location[1], location[0], round(value,2), ha='center', va='center') # Place it on graph
    plt.show()

  def draw_deterministic_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple deterministic policies
    input: Policies {np.array of np.array} -- array of policies to draw (each should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Policies)): # Go through all policies
      ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each policy
      ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
      for state, action in enumerate(Policies[subplot]):
        if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
          continue
        arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
        action_arrow = arrows[action] # Take the corresponding action
        location = self.locations[state] # Compute its location on graph
        plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
      ax.title.set_text(title[subplot]) # Set the title for the graph given as argument
    plt.show()

  def draw_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policies (draw an arrow in the most probable direction)
    input: Policy {np.array} -- array of policies to draw as probability
    output: /
    """
    deterministic_policies = np.array([[np.argmax(Policy[row,:]) for row in range(Policy.shape[0])] for Policy in Policies])
    self.draw_deterministic_policy_grid(deterministic_policies, title, n_columns, n_lines)

  def draw_value_grid(self, Values, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policy values
    input: Values {np.array of np.array} -- array of policy values to draw
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Values)): # Go through all values
      ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each value
      ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
      for state, value in enumerate(Values[subplot]):
        if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
          continue
        location = self.locations[state] # Compute the value location on graph
        plt.text(location[1], location[0], round(value,1), ha='center', va='center') # Place it on graph
      ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
    plt.show()


# ## Maze class

# In[4]:


# This class define the Maze environment

class Maze(object):

  # [Action required]
  def __init__(self):
    """
    Maze initialisation.
    input: /
    output: /
    """
    
    # [Action required]
    # Properties set from the CID
    y = 4; z = 8
    self._prob_success = 0.8 + 0.02*(9-y) # float
    # self._prob_success = 0.25
    self._gamma = 0.8 + 0.02*y # float
    # self._gamma = 0.8
    self._goal = 0 # integer (0 for R0, 1 for R1, 2 for R2, 3 for R3)

    # Build the maze
    self._build_maze()
                              

  # Functions used to build the Maze environment 
  # You DO NOT NEED to modify them
  def _build_maze(self):
    """
    Maze initialisation.
    input: /
    output: /
    """

    # Properties of the maze
    self._shape = (13, 10)
    self._obstacle_locs = [
                          (1,0), (1,1), (1,2), (1,3), (1,4), (1,7), (1,8), (1,9), \
                          (2,1), (2,2), (2,3), (2,7), \
                          (3,1), (3,2), (3,3), (3,7), \
                          (4,1), (4,7), \
                          (5,1), (5,7), \
                          (6,5), (6,6), (6,7), \
                          (8,0), \
                          (9,0), (9,1), (9,2), (9,6), (9,7), (9,8), (9,9), \
                          (10,0)
                         ] # Location of obstacles
    self._absorbing_locs = [(2,0), (2,9), (10,1), (12,9)] # Location of absorbing states
    self._absorbing_rewards = [ (500 if (i == self._goal) else -50) for i in range (4) ]
    self._starting_locs = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9)] #Reward of absorbing states
    self._default_reward = -1 # Reward for each action performs in the environment
    self._max_t = 500 # Max number of steps in the environment

    # Actions
    self._action_size = 4
    self._direction_names = ['N','E','S','W'] # Direction 0 is 'N', 1 is 'E' and so on
        
    # States
    self._locations = []
    for i in range (self._shape[0]):
      for j in range (self._shape[1]):
        loc = (i,j) 
        # Adding the state to locations if it is no obstacle
        if self._is_location(loc):
          self._locations.append(loc)
    self._state_size = len(self._locations)

    # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
    self._neighbours = np.zeros((self._state_size, 4)) 
    
    for state in range(self._state_size):
      loc = self._get_loc_from_state(state)

      # North
      neighbour = (loc[0]-1, loc[1]) # North neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('N')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('N')] = state

      # East
      neighbour = (loc[0], loc[1]+1) # East neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('E')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('E')] = state

      # South
      neighbour = (loc[0]+1, loc[1]) # South neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('S')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('S')] = state

      # West
      neighbour = (loc[0], loc[1]-1) # West neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('W')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('W')] = state

    # Absorbing
    self._absorbing = np.zeros((1, self._state_size))
    for a in self._absorbing_locs:
      absorbing_state = self._get_state_from_loc(a)
      self._absorbing[0, absorbing_state] = 1

    # Transition matrix
    self._T = np.zeros((self._state_size, self._state_size, self._action_size)) # Empty matrix of domension S*S*A
    for action in range(self._action_size):
      for outcome in range(4): # For each direction (N, E, S, W)
        # The agent has prob_success probability to go in the correct direction
        if action == outcome:
          prob = 1 - 3.0 * ((1.0 - self._prob_success) / 3.0) # (theoritically equal to self.prob_success but avoid rounding error and garanty a sum of 1)
        # Equal probability to go into one of the other directions
        else:
          prob = (1.0 - self._prob_success) / 3.0
          
        # Write this probability in the transition matrix
        for prior_state in range(self._state_size):
          # If absorbing state, probability of 0 to go to any other states
          if not self._absorbing[0, prior_state]:
            post_state = self._neighbours[prior_state, outcome] # Post state number
            post_state = int(post_state) # Transform in integer to avoid error
            self._T[prior_state, post_state, action] += prob

    # Reward matrix
    self._R = np.ones((self._state_size, self._state_size, self._action_size)) # Matrix filled with 1
    self._R = self._default_reward * self._R # Set default_reward everywhere
    for i in range(len(self._absorbing_rewards)): # Set absorbing states rewards
      post_state = self._get_state_from_loc(self._absorbing_locs[i])
      self._R[:,post_state,:] = self._absorbing_rewards[i]

    # Creating the graphical Maze world
    self._graphics = GraphicsMaze(self._shape, self._locations, self._default_reward, self._obstacle_locs, self._absorbing_locs, self._absorbing_rewards, self._absorbing)
    
    # Reset the environment (??????)
    self.reset()


  def _is_location(self, loc):
    """
    Is the location a valid state (not out of Maze and not an obstacle)
    input: loc {tuple} -- location of the state
    output: _ {bool} -- is the location a valid state
    """
    if (loc[0] < 0 or loc[1] < 0 or loc[0] > self._shape[0]-1 or loc[1] > self._shape[1]-1):
      return False
    elif (loc in self._obstacle_locs):
      return False
    else:
      return True


  def _get_state_from_loc(self, loc):
    """
    Get the state number corresponding to a given location
    input: loc {tuple} -- location of the state
    output: index {int} -- corresponding state number
    """
    return self._locations.index(tuple(loc))


  def _get_loc_from_state(self, state):
    """
    Get the state number corresponding to a given location
    input: index {int} -- state number
    output: loc {tuple} -- corresponding location
    """
    return self._locations[state]

  # Getter functions used only for DP agents
  # You DO NOT NEED to modify them
  def get_T(self):
    return self._T

  def get_R(self):
    return self._R

  def get_absorbing(self):
    return self._absorbing

  # Getter functions used for DP, MC and TD agents
  # You DO NOT NEED to modify them
  def get_graphics(self):
    return self._graphics

  def get_action_size(self):
    return self._action_size

  def get_state_size(self):
    return self._state_size

  def get_gamma(self):
    return self._gamma

  # Functions used to perform episodes in the Maze environment
  def reset(self):
    """
    Reset the environment state to one of the possible starting states
    input: /
    output: 
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """
    self._t = 0
    self._state = self._get_state_from_loc(self._starting_locs[random.randrange(len(self._starting_locs))])
    self._reward = 0
    self._done = False
    return self._t, self._state, self._reward, self._done

  def step(self, action):
    """
    Perform an action in the environment
    input: action {int} -- action to perform
    output: 
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """

    # If environment already finished, print an error
    if self._done or self._absorbing[0, self._state]:
      print("Please reset the environment")
      return self._t, self._state, self._reward, self._done

    # Drawing a random number used for probaility of next state
    probability_success = random.uniform(0,1)

    # Look for the first possible next states (so get a reachable state even if probability_success = 0)
    new_state = 0
    while self._T[self._state, new_state, action] == 0: 
      new_state += 1
    assert self._T[self._state, new_state, action] != 0, "Selected initial state should be probability 0, something might be wrong in the environment."

    # Find the first state for which probability of occurence matches the random value
    total_probability = self._T[self._state, new_state, action]
    while (total_probability < probability_success) and (new_state < self._state_size-1):
     new_state += 1
     total_probability += self._T[self._state, new_state, action]
    assert self._T[self._state, new_state, action] != 0, "Selected state should be probability 0, something might be wrong in the environment."
    
    # Setting new t, state, reward and done
    self._t += 1
    self._reward = self._R[self._state, new_state, action]
    self._done = self._absorbing[0, new_state] or self._t > self._max_t
    self._state = new_state
    return self._t, self._state, self._reward, self._done


# ## DP Agent

# In[5]:


# This class define the Dynamic Programing agent 
# The code is taken from lab 2 solution
class DP_agent(object):
 def value_iteration(self, env, gamma, threshold = 0.0001):
    R = env.get_R() # reward (np.array) (state_size x state_size x action_size)
    T = env.get_T() # transition matrix (np.array) (state_size x state_size x action_size)

    state_size = env.get_state_size() # number of states (int)
    action_size = env.get_action_size() # number of actions (int)
    absorbing = env.get_absorbing() # absorbing states label 1/0 (np.array)(1x state_size)

    # Initialisation
    epochs = 0
    delta = threshold # Setting value of delta to go through the first breaking condition
    V = np.zeros(state_size) # Initialise values at 0 for each state

    while delta >= threshold:
      epochs += 1 # Increment the epoch
      delta = 0 # Reinitialise delta value

      # For each state
      for prior_state in range(state_size):

        # If not an absorbing state
        if not absorbing[0, prior_state]:
                  
          # Store the previous value for that state
          v = V[prior_state] 

          # Compute Q value
          Q = np.zeros(4) # Initialise with value 0
          for post_state in range(state_size):
            Q += T[prior_state, post_state,:] * (R[prior_state, post_state, :] + gamma * V[post_state])
                
          # Set the new value to the maximum of Q
          V[prior_state]= np.max(Q) 

          # Compute the new delta
          delta = max(delta, np.abs(v - V[prior_state]))
            
            
    # When the loop is finished, fill in the optimal policy
    policy = np.zeros((state_size, action_size)) # Initialisation

    for prior_state in range(state_size):
      # Compute the Q value
      Q = np.zeros(4)
      for post_state in range(state_size):
        Q += T[prior_state, post_state,:] * (R[prior_state, post_state, :] + gamma * V[post_state])
            
      # The action that maximises the Q value gets probability 1
      policy[prior_state, np.argmax(Q)] = 1 

    return policy, V, epochs

  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
 def solve(self, env):
    """
    Solve a given Maze environment using Dynamic Programming
    input: env {Maze object} -- Maze to solve
    output: 
      - policy {np.array} -- Optimal policy found to solve the given Maze environment 
      - V {np.array} -- Corresponding value function 
    """

    #### 
    # Add your code here
    # WARNING: for this agent only, you are allowed to access env.get_T(), env.get_R() and env.get_absorbing()
    ####
    gamma = env.get_gamma()
    optimal_policy, optimal_value, epoch = self.value_iteration(env, gamma, threshold = 0.0001)

    return optimal_policy, optimal_value


# ## MC agent

# In[6]:


from matplotlib.widgets import EllipseSelector
# This class define the Monte-Carlo agent

class MC_agent(object):
  
  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
  def walk(self, policy, env):
    """
    Record trace and reward of the MC agent in one episode given a policy
    input:
      policy {np.array} -- policy given to walk in given Maze environment (state_size x action_size)
      env {Maze object} -- Maze to walk
    Output:
      states_visited {list} -- list of states visited in this episode
      rewards {list} -- reward along the walk in this episode
      actions_taken {list} --action taken along the walk in this episode
    """
    walk_done = False
    states_visited = []
    rewards = []
    actions_taken = []
    t, current_state, reward, walk_done = env.reset()
    # keep walking until absorbing states or reach max step
    while (not walk_done) and t<3000:
      action = np.random.choice(range(env.get_action_size()), p=policy[current_state,:])
      states_visited.append(current_state)
      actions_taken.append(action)

      t, current_state, reward, walk_done = env.step(action)
      rewards.append(reward)
    return states_visited, rewards, actions_taken
    
 
  def update_Q(self, old_Q, trace, rewards, actions_taken, env, N):
    """
    Update action-value function according to agent's experience
    input:
      old_Q {np.array}-- old action-value function (state_size x action_size)
      trace {list}-- agent's trace in a walk
      rewards {list}-- agent's reward experience along the walk
      actions_taken {list}-- agent's action along the walk
      env {Maze object} -- Maze to solve
      N {np.array} -- table to store number of times visit state s through action a
    output: 
      new_Q {np.array}-- new action-value function (state_size x action_size)
      N {np.array} -- table to store number of times visit state s through action a ##reset n
    """
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    gamma = env.get_gamma()
    # N = np.zeros((env.get_state_size(), env.get_action_size()))

    new_Q = np.copy(old_Q)

    # agent experience
    number_of_steps = len(trace)

    state_action_visited = []

    # update Q according to experience
    for i in range(number_of_steps):
      ith_state = trace[i]
      ith_action = actions_taken[i]
      if (ith_state, ith_action) in state_action_visited:
        continue
      else:
        gain = 0
        for reward_index, reward in enumerate(rewards[i:]):
          gain = gain + reward * (gamma ** reward_index)
        # use first time MC update
        N[ith_state, ith_action]+=1
        new_Q[ith_state, ith_action] = old_Q[ith_state, ith_action] + 1/N[ith_state, ith_action] *(gain - old_Q[ith_state, ith_action])
        state_action_visited.append((ith_state, ith_action))
    return new_Q, N


  def update_policy(self, Q, epsilon, env, k):
    """
    Epsilon-greedy policy improvement
    input:
      Q {np.array}-- action-value function (state_size x action_size)
      epsilon {float}-- probability to choose action at random 
      env {Maze object} -- Maze to solve
      k {int}-- number of update made to discount epsilon
    output: 
      policy {np.array}-- new policy according to given Q
    """
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    policy = np.zeros((state_size, action_size))

    discounted_epsilon = epsilon *(0.9995**k)
    # update policy
    for state in range(state_size):
      optimal_action = np.argmax(Q[state, :])
      for action in range(action_size):
        if action==optimal_action:
          policy[state][action] = 1-discounted_epsilon+(discounted_epsilon/env.get_action_size())
        else:
          policy[state][action] = discounted_epsilon/env.get_action_size()

    return policy
  

  def solve(self, env):
    """
    Solve a given Maze environment using Monte Carlo learning
    input: env {Maze object} -- Maze to solve
    output: 
      - policy {np.array} -- Optimal policy found to solve the given Maze environment 
      - values {list of np.array} -- List of successive value functions for each episode 
      - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
    """

    # Initialisation (can be edited)
    Q = np.random.rand(env.get_state_size(), env.get_action_size())
    V = np.zeros(env.get_state_size())
    N = np.zeros((env.get_state_size(), env.get_action_size())) # count how many times visit state s through action a
    policy = np.zeros((env.get_state_size(), env.get_action_size())) 
    values = []
    total_rewards = []
    k = 0
    epsilon = 0.4
    policy = self.update_policy(Q, epsilon, env, 1)

    while k<7000:
      k+=1
      states_visited, rewards, actions_taken = self.walk(policy, env)
      Q, N = self.update_Q(old_Q=Q, trace=states_visited, rewards=rewards, actions_taken=actions_taken, env=env, N=N)
      policy = self.update_policy(Q, epsilon, env, k)
      v = np.max(Q,1)
      values.append(v)
      total_rewards.append(sum(rewards))
    return policy, values, total_rewards


# In[ ]:





# ## TD agent

# In[7]:


# This class define the Temporal-Difference agent

class TD_agent(object):

  def update_Q(self, old_Q, alpha, gamma, current_state, action_taken, reward, next_state):
    """
    Update Q according to Q-learning
    input: 
      old_Q {np.array} -- action-value function to be updated
      alpha {float} -- discount rate of q-value for next episode
      gamma {float} -- discount rate of q-value for next step
      current_state {int} -- agent's current state
      action_taken {int} -- agent's action taken at current state
      next_state {int} -- agent's next state resulting from action_taken
    output:
      new_Q {np.array} -- updated action-value function
    """
    new_Q = np.copy(old_Q)
    optimal_action = np.argmax(old_Q[next_state,:])
    new_Q[current_state, action_taken] = old_Q[current_state, action_taken] + alpha*(reward+gamma*old_Q[next_state,optimal_action]-old_Q[current_state, action_taken])

    return new_Q

  def update_policy(self, Q, epsilon, env, k):
    """
    Epsilon-greedy policy improvement
    input:
    Q {np.array}-- action-value function (state_size x action_size)
    epsilon {float}-- probability to choose action at random 
    env {Maze object} -- Maze to solve
    k {int}-- number of update made to discount epsilon
    output: 
    policy {np.array}-- new policy according to given Q
    """
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    policy = np.zeros((state_size, action_size))

    discounted_epsilon = epsilon*(0.9995**k)
    # update policy
    for state in range(state_size):
      optimal_action = np.argmax(Q[state, :])
      for action in range(action_size):
        if action==optimal_action:
          policy[state][action] = 1-discounted_epsilon+(discounted_epsilon/env.get_action_size())
        else:
          policy[state][action] = discounted_epsilon/env.get_action_size()

    return policy


  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
  def solve(self, env):
    """
    Solve a given Maze environment using Temporal Difference learning
    input: env {Maze object} -- Maze to solve
    output: 
      - policy {np.array} -- Optimal policy found to solve the given Maze environment 
      - values {list of np.array} -- List of successive value functions for each episode 
      - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
    """

    # Initialisation (can be edited)
    Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
    V = np.zeros(env.get_state_size())
    policy = np.zeros((env.get_state_size(), env.get_action_size())) 
    values = []
    total_rewards = []
    alpha = 0.1
    gamma = env.get_gamma()
    epsilon = 0.4
    k = 0
    action_size = env.get_action_size()
    state_size = env.get_state_size()

    while k<7000:
      k+=1
      t, current_state, reward, walk_done = env.reset()
      policy = self.update_policy(Q, epsilon, env, k)
      reward_along_episode = []
      while (not walk_done) and t<3000:
        action = np.random.choice(range(action_size), p=policy[current_state,:])
        t, next_state, reward, walk_done = env.step(action)
        reward_along_episode.append(reward)
        Q = self.update_Q(old_Q=Q, alpha=alpha, gamma=gamma, current_state=current_state, action_taken=action, reward=reward, next_state=next_state)
        policy = self.update_policy(Q, epsilon, env, k)
        current_state = next_state
      v = np.max(Q,1)
      values.append(v)
      total_rewards.append(sum(reward_along_episode))
      
    
    return policy, values, total_rewards


# ## Example main

# In[10]:


if __name__=="__main__":
    ### Question 0: Defining the environment

    print("Creating the Maze:\n")
    maze = Maze()

    ### Question 1: Dynamic programming

    dp_agent = DP_agent()
    dp_policy, dp_value = dp_agent.solve(maze)

    #print("Results of the DP agent:\n")
    #maze.get_graphics().draw_policy(dp_policy)
    #maze.get_graphics().draw_value(dp_value)


    ### Parameter tracker
    n_run = 1
    n_episodes = 7000
    epsilon = 0.4
    alpha = 0.1


    ### Question 2: Monte-Carlo learning
    #########################################################################
    mc_agent = MC_agent()

    mse_mc_value = np.zeros((n_run, n_episodes))
    n_total_mc_rewards = np.zeros((n_run, n_episodes))

    for run in range(n_run):
        mc_policy, mc_values, total_mc_rewards = mc_agent.solve(maze)
        for idx_episode, values in enumerate(mc_values):
            mse_mc_value[run, idx_episode] = mean_squared_error(dp_value, values, squared=True)  # calculate mse for each point
        n_total_mc_rewards[run] = total_mc_rewards

    # print value and policy
    print("Results of the MC agent:\n")
    #maze.get_graphics().draw_policy(mc_policy)
    #maze.get_graphics().draw_value(mc_values[-1])

    # Q2 x and y values for plots
    x = [i for i in range(1, n_episodes+1)]
    y = np.mean(n_total_mc_rewards, axis=0)
    y_std = np.std(n_total_mc_rewards, axis=0)

    # Q2 plot for mean and std deviation graphs
    plt.plot(x, y, label="mean", color='purple')
    plt.fill_between(x, y-y_std, y+y_std, label="std deviation", alpha=0.4, facecolor="blue")
    plt.legend(loc="lower right")
    plt.xlabel("No.of Episodes")  
    plt.ylabel("Total Non-discounted Sum of Rewards")  
    plt.show()



    ### Question 3: Temporal-Difference learning
    #########################################################################
    td_agent = TD_agent()

    n_total_td_rewards = np.zeros((n_run, n_episodes))
    mse_td_value = np.zeros((n_run, n_episodes))

    for run in range(n_run):
        td_policy, td_values, total_td_rewards = td_agent.solve(maze)
        for idx_episode, values in enumerate(td_values):
            mse_td_value[run, idx_episode] = mean_squared_error(dp_value, values, squared=True)   # calculate mse for each point
        n_total_td_rewards[run] = total_td_rewards

    # print value and policy
    #print("Results of the TD agent:\n")
    #maze.get_graphics().draw_policy(td_policy)
    #maze.get_graphics().draw_value(td_values[-1])

    # Q3 x and y values
    x = [i for i in range(1, n_episodes+1)]
    y = np.mean(n_total_td_rewards, axis=0)
    y_std = np.std(n_total_td_rewards, axis=0)

    # Q3 plot for mean and std deviation graphs
    plt.plot(x, y, label="mean", color='orange')
    plt.fill_between(x, y-y_std, y+y_std, label="std deviation", alpha=0.4, facecolor="green")
    plt.legend(loc="lower right")
    plt.xlabel("No.of Episodes")  
    plt.ylabel("Total Non-discounted Sum of Rewards")  
    plt.show()


    ### Q4 Comparasion of learner ###
    #########################################################################
    # part 1
    # x and y values & plots for TD & MC
    mse_x = [i for i in range(1, n_episodes+1)]
    mse_td_y = np.mean(mse_td_value, axis=0)
    mse_td_std = np.std(mse_td_value, axis=0)
    mse_mc_y = np.mean(mse_mc_value, axis=0)
    mse_mc_std = np.std(mse_mc_value, axis=0)

    plt.plot(mse_x, mse_td_y, label="TD mean", color='orange')
    plt.fill_between(mse_x, mse_td_y-mse_td_std, mse_td_y+mse_td_std, label="TD std deviation", alpha=0.5, facecolor="green")
    plt.plot(mse_x, mse_mc_y, label="MC mean", color='purple')
    plt.fill_between(mse_x, mse_mc_y-mse_mc_std, mse_mc_y+mse_mc_std, label="MC std deviation", alpha=0.5, facecolor="blue")
    plt.legend(loc="upper right")
    plt.xlabel("No.of Episodes") 
    plt.ylabel("MSE of Optimal Value Function")  
    plt.show()

    # Q4 part 3 - scatter plot
    for run in range(n_run):
        y_td_total_reward = n_total_td_rewards[run]
        x_td_fn_error = mse_td_value[run]
        y_mc_total_reward = n_total_mc_rewards[run]
        x_mc_fn_error = mse_mc_value[run]

        plt.scatter(x_mc_fn_error, y_mc_total_reward, c='purple', marker='s', label='MC')
        plt.scatter(x_td_fn_error, y_td_total_reward, c='orange', marker='o', label='TD')
        plt.xlabel("MSE of State Value")  
        plt.ylabel("Total Non-Discounted Sum of Reward") 
        plt.legend(loc='lower left')
        plt.show()



# In[ ]:





# In[ ]:





# In[ ]:




