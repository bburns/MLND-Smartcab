
"""
Smartcab Learning Agent
"""

import random
import math

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """
    An agent that learns to drive in the Smartcab world.
    This is the object you will be modifying.
    """

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        """
        learning - Whether the agent is expected to learn
        epsilon  - Random exploration factor, a probability, 0.0-1.0
        alpha    - Learning rate, 0.0-1.0
        """
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        self.coverage = 0        # Percent of state-action space covered

        ###########
        ## TO DO ##
        ###########
        # Define the tuple of states that you want to use, the number of
        # possible states, and the number of possible state-action values.
        # These are used in constructing the state object in the build_state
        # method and calculating the percent of the state-action space covered,
        # for each time step.
        self.states = ()          # e.g. ('light','oncoming')
        self.n_states = 0         # e.g. 2 * 4
        self.n_state_actions = 0  # e.g. self.n_states * number of actions available

        # e.g. this is what I used -
        self.states = ('light','waypoint','oncoming','left')
        self.n_states = 2 * 3 * 4 * 4  # 96
        self.n_state_actions = self.n_states * len(self.valid_actions)  # 384

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.trial = 1      # track trial number
        self.n_trials = 10  # default number of trials, override in run fn



    def reset(self, destination=None, testing=False):
        """
        Reset the agent's state.
        Called at the beginning of each trial.
        'testing' is set to True if testing trials are being used
        once training trials have completed.
        """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)

        ###########
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.epsilon = 0.0 # no random exploration
            self.alpha = 0.0   # no learning
        else:
            # unoptimized
            # self.epsilon = self.epsilon - 0.05

            # optimized
            trial = self.trial
            frac = float(trial)/self.n_trials
            frac0 = float(trial-1)/self.n_trials
            # self.epsilon = 1-frac0 # linear 1 to 0
            # self.epsilon = 0.9**trial
            # self.epsilon = 1.0 / trial
            # self.epsilon = 1.0 / trial**2
            # self.epsilon = math.exp(-0.1*(trial-1))
            self.epsilon = math.cos(math.pi/2 * frac)

            # make sure we stay in bounds 0 to 1
            if self.epsilon > 1.0:
                self.epsilon = 1.0
            elif self.epsilon < 0.0:
                self.epsilon = 0.0

            # adjust learning rate alpha
            # self.alpha = 1.0 if frac < 0.8 else 0.1
            # self.alpha = self.epsilon
            self.alpha = 1-frac0 # linear 1 to 0

            self.trial += 1


    def build_state(self):
        """
        Build a state object for the agent.
        Called by the agent when it requests data from the environment.
        The next waypoint, the intersection inputs, and the deadline
        are all features available to the agent.
        """

        # Collect data about the environment
        # waypoint = self.planner.next_waypoint() # The next waypoint
        # inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        # deadline = self.env.get_deadline(self)  # Remaining deadline #. remaining time

        # Collect all information available into one dictionary,
        # so user can just supply state names in agent initialization.
        senses = self.env.sense(self)
        inputs = {
            'waypoint': self.planner.next_waypoint(),
            'light':    senses['light'],
            'oncoming': senses['oncoming'],
            'left':     senses['left'],
            'right':    senses['right'],
            'deadline': self.env.get_deadline(self),
        }

        # Build a state object using the states specified in agent initialization.
        # e.g. if self.states was set to ('light', 'oncoming') in __init__,
        # the state object might be something like ('red', 'forward').
        # Note: state must be a hashable type to use in a dictionary, so use a tuple, not a list.
        state = tuple([inputs[state_name] for state_name in self.states])

        # this is already done in the update method
        # ###########
        # ## TO DO ##
        # ###########
        # # When learning, check if the state is in the Q-table
        # #   If it is not, create a dictionary in the Q-table for the current 'state'
        # #   For each action, set the Q-value for the state-action pair to 0
        # if self.learning:
        #     self.createQ(state) # create state, if needed

        return state


    def get_maxQ(self, state):
        """
        Find the maximum Q-value of all actions based on the 'state' the smartcab is in.
        This is called by the agent when it is deciding which action to take.
        """

        ###########
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        # maxQ = None
        d = self.Q[state]
        maxQ = max(d.values())
        return maxQ


    def createQ(self, state):
        """
        Create a new Q-table.
        Called when a state is generated by the agent.
        """

        ###########
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to None
        if self.learning:
            if not state in self.Q:
                data = dict()
                #. use self.states eg
                # for state_name in self.states:
                    # data[state_name] = None
                data[None]      = None
                data['forward'] = None
                data['right']   = None
                data['left']    = None
                self.Q[state] = data


    def choose_action(self, state):
        """
        Choose an action to take based on the 'state' the smartcab is in.
        Called by the agent's update method.
        """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        # action = None

        ###########
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        if not self.learning:
            action = random.choice(self.valid_actions)
        else:
            # This is called the Epsilon-Greedy Selection method
            data = self.Q[state]
            if random.random() < self.epsilon:
                # choose from actions with None for reward value
                actions = []
                for action in data:
                    if data[action] is None:
                        actions.append(action)
                if actions:
                    action = random.choice(actions)
                else:
                    action = random.choice(self.valid_actions)
            else:
                # Choose an action with highest Q-value for the current state.
                # Note: ties should be resolved randomly.
                # maxQ = -9e9
                # for action in data:
                    # value = data[action] or 0
                    # if value > maxQ:
                        # maxQ = value
                # find maxQ value
                maxQ = max(data.values()) or 0
                # check for ties
                actions = []
                for action in data:
                    value = data[action] or 0
                    if value == maxQ:
                        actions.append(action)
                # pick an action from max value actions
                action = random.choice(actions)

        return action


    def learn(self, state, action, reward):
        """
        Update the Q matrix based on the completed action and reward.
        Called after the agent completes an action and receives an award.
        This function does not consider future rewards when conducting learning.
        """

        ###########
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha   ' (do not use the discount factor 'gamma')

        # The q-learning algorithm should similar to the following format:
        # qnew = (1-alpha)*qold + alpha*[reward + gamma * qmax]

        # self.Q[state][action] = reward # this worked fairly well - ie alpha always =1
        Q = self.Q[state][action] or 0
        Qnew = (1-self.alpha)*Q + self.alpha * reward
        self.Q[state][action] = Qnew


    def update(self):
        """
        Update the agent for one time step.
        Build the agent state, choose an action, receive a reward, and learn if enabled.
        """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table, if needed
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward #. should be reward = self.act(action)
        self.learn(state, action, reward)   # Q-learn

        # Calculate coverage of state-action space
        #. do this elsewhere?
        n_seen = 0
        for rewards in self.Q.values():
            for reward in rewards.values():
                if not reward is None:
                    n_seen += 1
        self.coverage = float(n_seen) / self.n_state_actions


def run():
    """
    Run the simulation.
    Press [ESC] to close the simulation, or [SPACE] to pause the simulation.
    """
    #. shouldn't run be in sim? ie creating environment and agent etc?

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()

    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #   epsilon    - continuous value for the exploration factor, default is 1
    #   alpha      - continuous value for the learning rate, default is 0.5
    # agent = env.create_agent(LearningAgent)
    # agent = env.create_agent(LearningAgent, learning=True)
    agent = env.create_agent(LearningAgent, learning=True, alpha=0.5)
    # agent = env.create_agent(LearningAgent, learning=True, alpha=0.8)

    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    # env.set_primary_agent(agent)
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    # sim = Simulator(env)
    # sim = Simulator(env, update_delay=0, log_metrics=True, display=False) # use for unoptimized dataset
    sim = Simulator(env, update_delay=0, log_metrics=True, optimized=True, display=False)

    ##############
    # Run the simulator
    # Flags:
    #   tolerance    - epsilon tolerance before beginning testing, default is 0.05
    #   n_test       - discrete number of testing trials to perform, default is 0
    agent.n_trials = 250
    sim.run(n_test=40)


if __name__ == '__main__':
    run()
