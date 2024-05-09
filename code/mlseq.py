import numpy as np
from utilities import Utilities
from mllib import MLUtilities


#################################
# State Machine
#################
class StateMachine(MLUtilities,Utilities):
    """ Basic building block for state machine. """
    def __init__(self,verbose=True,logfile=None,initial_state=None):
        self.verbose = verbose
        self.logfile = logfile

        self.initial_state = initial_state

    def transition_func(self,s,x):
        """ s: previous state, x: current input. Returns current state. """
        raise NotImplementedError
    
    def output_func(self,s):
        """ s: current state. Returns current output. """
        raise NotImplementedError
        
    def transduce(self,input_seq):
        """ Transduce state machine for given input and initialisation."""
        s = self.initial_state
        out = []
        for x in input_seq:
            s = self.transition_func(s,x)
            out.append(self.output_func(s))
        return np.array(out)
        
#################################

#################################
# Accumulator as simple state machine
#################
class Accumulator(StateMachine):
    def __init__(self):
        StateMachine.__init__(self,initial_state=0)
    
    def transition_func(self,s,x):
        return s + x
    
    def output_func(self,s):
        return s
#################################


#################################
# Binary addition
#################
class BinaryAddition(StateMachine):
    def __init__(self):
        StateMachine.__init__(self,initial_state=[0,0]) # s = (value,carry flag)
    
    def transition_func(self,s,x):
        val = x[0] + x[1] + s[1]
        s[0] = val % 2
        s[1] = 1 if val > 1 else 0
        return s
    
    def output_func(self,s):
        return s[0]
#################################

#################################
# Reverser
#################
class Reverser(StateMachine):
    def __init__(self):
        StateMachine.__init__(self,initial_state=[None,[],1]) # s = (current output,stored values,flag)

    def transition_func(self,s,x):
        if (x != 'end') & s[2]:
            s[0] = None
            s[1].append(x)
        else:
            s[2] = 0
            if len(s[1]) > 0:
                s[0] = s[1].pop(-1)
            else:
                s[0] = None
        return s
    
    def output_func(self,s):
        return s[0]
        
#################################

#################################
# Recurrent neural network
#################
class RecurrentNeuralNetwork(StateMachine):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, initial_state=None):
        self.Wsx = Wsx
        self.Wss = Wss
        self.Wo = Wo
        self.Wss_0 = Wss_0
        self.Wo_0 = Wo_0
        self.f1 = f1
        self.f2 = f2
        self.s_dim = self.Wsx.shape[0]
        self.x_dim = self.Wsx.shape[1]
        self.initial_state = np.zeros((self.s_dim,1))
        StateMachine.__init__(self,initial_state=self.initial_state)

    def transition_func(self,s,x):
        linear = np.dot(self.Wss,s) + np.dot(self.Wsx,x) + self.Wss_0
        return self.f1(linear)

    def output_func(self,s):
        linear = np.dot(self.Wo,s) + self.Wo_0
        return self.f2(linear)


#################################

#################################
# Markov Decision Process
#################
class MarkovDecisionProcess(MLUtilities,Utilities):
    """ Markov decision process. """
    def __init__(self,states,transition,reward,verbose=True,logfile=None):
        """ Markov decision process. 
            -- states: list of possible states
            -- transition: dictionary of transition matrices, whose keys will be stored as self.actions
            -- reward: function object with call sign reward(state,action). Must be compatible with states and transition.
        """
        self.verbose = verbose
        self.logfile = logfile
        self.states = np.array(states)
        self.transition = transition
        self.actions = np.array(list(self.transition.keys()))
        self.reward = np.vectorize(reward)

    def value_func(self,policy,horizon=None,gamma=0.1):
        """ Calculate (finite horizon) value for given policy. 
            -- policy: function object mapping states to actions. 
            -- horizon: None or non-negative integer. 
                        If None (default), then calculate infinite-horizon result with discount gamma. 
            -- gamma: discount value in (0,1). Only used if horizon is None.
            Returns array of values of shape (len(self.states),).
        """
        if horizon is None:
            if (gamma < 0.0) | (gamma > 1.0):
                raise ValueError("Discount gamma should be between zero and unity for infinite-horizon value function.")
        vec_pol = np.vectorize(policy)
        policy_actions = vec_pol(self.states)
        if np.any(self.select_not_these(policy_actions,set(self.actions))):
            raise ValueError("Unexpected action detected. Try with another policy.")

        policy_rewards = self.reward(self.states,policy_actions)
        
        trans_matrix = np.zeros((len(self.states),len(self.states)))
        for s in range(len(self.states)):
            trans_matrix[s] = self.transition[policy_actions[s]][s] # set each row

        values = np.zeros(len(self.states))
        if horizon is None:
            # careful with inversion
            values = np.dot(np.linalg.inv(np.eye(len(self.states)) - gamma*trans_matrix),policy_rewards)
        else:
            for h in range(1,horizon+1):
                values = policy_rewards + np.dot(trans_matrix,values)

        return values

    def value_iteration(self,horizon=None,gamma=1.0,eps=1e-3,max_iter=1000):
        """ Calculate (finite horizon) optimal action-value function Q^{h}(s,a) using value iteration.
            -- horizon: None or non-negative integer. 
                        If None (default), then calculate infinite-horizon result with discount gamma. 
            -- gamma: discount value in (0,1). Set to 1 if horizon is finite integer.
            -- eps: small positive float, controls stopping criterion for infinite-horizon case. Only used if horizon is None.
            -- max_iter: maximum number of iterations
            Returns array of values of shape (len(self.actions),len(self.states)).
        """
        if horizon is None:
            if (gamma < 0.0) | (gamma > 1.0):
                raise ValueError("Discount gamma should be between zero and unity for infinite-horizon value function.")
        else:
            gamma = 1.0
        Q = np.zeros((len(self.actions),len(self.states)))
        
        if horizon == 0:
            return Q
        
        rewards = np.zeros_like(Q)
        for a in range(len(self.actions)):
            rewards[a] = self.reward(self.states,self.actions[a])
            
        if horizon is None:
            Q_prev = Q.copy() # store previous Q values for exit condition
        else:
            h = 0 # counter for finite-horizon case
            
        for i in range(max_iter):
            Qmax = np.max(Q,axis=0)
            for a in range(len(self.actions)):
                Q[a] = rewards[a] + gamma*np.dot(self.transition[self.actions[a]],Qmax)
            if horizon is None:
                if np.max(np.fabs(Q - Q_prev)) < eps:
                    return Q
                Q_prev = Q.copy()
            else:
                if h == horizon-1:
                    return Q
                h += 1

        return Q

    def optimal_policy(self,horizon=None,gamma=1.0,eps=1e-3,max_iter=1000):
        Q = self.value_iteration(horizon=horizon,gamma=gamma,eps=eps,max_iter=max_iter)
        pol_vec = self.actions[np.argmax(Q,axis=0)]
        def opt_pol(state):
            s = np.where(state == self.states)[0][0]
            return pol_vec[s]
        return np.vectorize(opt_pol)
        
#################################
