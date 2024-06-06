import numpy as np
from utilities import Utilities
from mllib import MLUtilities,SeqUtilities
from mlalgos import Sequential

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
# Markov Decision Process (own code)
#################
class My_MarkovDecisionProcess(MLUtilities,Utilities):
    """ Markov decision process. """
    def __init__(self,states,transition,reward,verbose=True,logfile=None):
        """ Markov decision process. 
            -- states: list of possible states
            -- transition: dictionary of transition matrices, whose keys will be stored as self.actions
            -- reward: function object with call sign reward(state,action). Must be compatible with states and transition.
        """
        Utilities.__init__(self)
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
            -- gamma: discount value in (0,1). Expectimax search if horizon is finite integer.
            -- eps: small positive float, controls stopping criterion for infinite-horizon case. Only used if horizon is None.
            -- max_iter: maximum number of iterations
            Returns array of values of shape (len(self.actions),len(self.states)).
        """
        # if horizon is None:
        if (gamma < 0.0) | (gamma > 1.0):
            raise ValueError("Discount gamma should be between zero and unity.")
        # else:
        #     gamma = 1.0
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

#################################
# Markov Decision Process (concepts from MIT-OLL)
#################
class MarkovDecisionProcess(MLUtilities,Utilities,SeqUtilities):
    """ Markov decision process. """
    def __init__(self,states,actions,transition,reward,discount_factor=1.0,start_dist=None,verbose=True,logfile=None):
        """ Markov decision process. 
            -- states: list of possible states
            -- actions: list of possible actions
            -- transition: function from (state,action) into DDist over next state
            -- reward: function object with call sign reward(state,action). Must be compatible with states and transition.
            -- discount_factor: discount value in (0,1).
            -- start_dist: optional instance of DDist, specifying initial state dist.
                           If unspecified, set to uniform over states.
        """
        Utilities.__init__(self)
        self.verbose = verbose
        self.logfile = logfile
        self.states = states
        self.transition = transition
        self.actions = actions
        self.reward = reward
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist is not None else self.uniform_dist(states)


    def terminal(self, s):
        """ Given a state, return True if the state should be considered to
            be terminal (generates an infinite sequence of zero reward). """
        return False

    def init_state(self):
        """ Return an initial state by drawing from the distribution over start states."""
        return self.start.draw()

    def sim_transition(self, s, a):
        """ Simulates a transition from the given state, s and action a, using the
            transition model as a probability distribution.  If s is terminal,
            use init_state to draw an initial state.  Returns (reward, new_state). """
        return (self.reward(s, a),self.init_state() if self.terminal(s) else self.transition(s, a).draw())
    
    def state2vec(self, s):
        """ One-hot encoding of state s; used in neural network agent implementations. """
        v = np.zeros((1, len(self.states)))
        v[0,self.states.index(s)] = 1.
        return v.T # return v for keras, v.T for mlfundas
#################################

#################################
class TabularQ():
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])

    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy

    def set(self, s, a, v):
        self.q[(s,a)] = v

    def get(self, s, a):
        return self.q[(s,a)]

    def update(self,data,lrate): 
        one_m_lr = 1-lrate
        for sat in data:
            s,a,t = sat
            v = self.get(s,a)*one_m_lr + lrate*t
            self.set(s,a,v)
        return
#################################

#################################
class SeqLearn(SeqUtilities):
    """ Routines for various flavours of Q-learning. """
    #################################
    def QLearn(self,mdp,q,lrate=0.1,iters=100,eps=0.5,interactive=None):
        """ Learn Q function element-wise for given MDP.
            --   mdp: instance of MarkovDecisionProcess
            --     q: instance of TabularQ
            -- lrate: learning rate
            -- iters: max no. of iterations
            --   eps: exploration control for epsilon_greedy()
            Returns:
            -- updated q
        """
        # assume initialised Q provided as input
        # # initialise Q(s,a) = 0
        # for s in q.states:
        #     for a in q.actions:
        #         q.set(s,a,0.0)
        
        # pick starting state
        s = mdp.init_state()
        # learn
        for i in range(iters):
            # select action
            a = self.epsilon_greedy(q,s,eps=eps)
            # execute action
            r,s_pr = mdp.sim_transition(s,a)
            # now we have s,a,r,s_pr
            # calculate target
            future_val = 0.0 if mdp.terminal(s) else self.value(q,s_pr)
            t = r + mdp.discount_factor*future_val
            # update
            q.update([(s,a,t)],lrate)
            s = s_pr
            if interactive is not None: interactive(q,i)

        return q
    #################################


    #################################
    # courtesy MIT-OLL MLIntro Course
    def value_iteration(self,mdp, q, eps = 0.01, max_iters = 1000):
        def v(s):
            return self.value(q,s)
        for it in range(max_iters):
            new_q = q.copy()
            delta = 0
            for s in mdp.states:
                for a in mdp.actions:
                    new_q.set(s, a, mdp.reward_fn(s, a) + mdp.discount_factor*mdp.transition_model(s, a).expectation(v))
                    delta = max(delta, abs(new_q.get(s, a) - q.get(s, a)))
            if delta < eps:
                return new_q
            q = new_q
            
        return q
    #################################


    #################################
    # courtesy MIT-OLL MLIntro Course
    def sim_episode(self,mdp, episode_length, policy):
        """ Simulate an episode (sequence of transitions) of at most
            episode_length, using policy function to select actions.  If we find
            a terminal state, end the episode.  
            Return accumulated reward and a list
            of (s, a, r, s') where s' is None for transition from terminal state.
        """
        episode = []
        reward = 0
        s = mdp.init_state()
        for i in range(episode_length):
            a = policy(s)
            (r, s_prime) = mdp.sim_transition(s, a)
            reward += r
            if mdp.terminal(s):
                episode.append((s, a, r, None))
                break
            episode.append((s, a, r, s_prime))
            s = s_prime

        return reward, episode
    #################################


    #################################
    def QLearn_Batch(self,mdp,q,lrate=0.1,iters=100,eps=0.5,episode_length=10,n_episodes=2,interactive=None):
        """ Learn Q function for given MDP using episodes.
            --   mdp: instance of MarkovDecisionProcess
            --     q: instance of TabularQ
            -- lrate: learning rate
            -- iters: max no. of iterations
            --   eps: exploration control for epsilon_greedy()
            -- episode_length,n_episodes: self-explanatory.
            Returns:
            -- updated q
        """
        # initialise Q(s,a) = 0
        for s in q.states:
            for a in q.actions:
                q.set(s,a,0.0) 

        policy = lambda s: self.epsilon_greedy(q,s,eps=eps)
        
        # learn
        all_experiences = []
        for i in range(iters):
            for n in range(n_episodes):
                # simulate episode
                reward,episode = self.sim_episode(mdp,episode_length,policy)
                all_experiences += episode

            all_q_targets = []
            for exp in all_experiences:
                s,a,r,s_pr = exp
                # calculate target
                future_val = 0.0 if s_pr is None else self.value(q,s_pr)
                t = r + mdp.discount_factor*future_val
                all_q_targets.append((s,a,t))

            # update
            q.update(all_q_targets,lrate)
            if interactive: interactive(q,i)

        return q
    #################################
    
class NNQ(MLUtilities,Utilities):
    def __init__(self):
        Utilities.__init__(self,states,actions,state2vec,num_hidden_layers,num_units,epochs=1,no_minibatch=True)
        self.actions = actions
        self.states = states
        self.epochs = epochs
        self.state2vec = state2vec
        self.models = {a:self.make_nn(len(self.states),num_hidden_layers, num_units) for a in self.actions}
        self.no_minibatch = no_minibatch

    def make_nn(self,state_dim, num_hidden_layers, num_units):
        """
        state_dim = (int) number of states
        num_hidden_layers = (int) number of fully connected hidden layers
        num_units = (int) number of dense relu units to use in hidden layers
        """
        params_setup = {'data_dim':state_dim,'L':num_hidden_layers+1,'adam':True,
                        'n_layer':[num_units]*num_hidden_layers+[1],'standardize':True,'reg_fun':'none',
                        'atypes':['relu']*num_hidden_layers+['lin'],'loss_type':'square'}
        model = Sequential(params_setup)
        return model
        # # use structure below for keras
        # model = Sequential()
        # model.add(Dense(num_units, input_dim = state_dim, activation='relu'))
        # for i in range(num_hidden_layers-1):
        #     model.add(Dense(num_units, activation='relu'))
        # model.add(Dense(1, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam())
        # return model

    def get(self, s, a):
        return self.models[a].predict(self.state2vec(s))

    # list selection trick courtesy MIT-OLL MLIntro Course
    def update(self, data, lr):
        for a in self.actions:
            if [s for (s, at, t) in data if a==at]:
                X = np.vstack([self.state2vec(s) for (s, at, t) in data if a==at]).T
                Y = np.vstack([t for (s, at, t) in data if a==at]).T
                mbc = Y.shape[1] if self.no_minibatch else int(np.sqrt(Y.shape[1]))
                self.models[ia].train(X,Y,params={'max_epoch':self.epochs,'lrate':lr,'check_after':epochs+1,'mb_count':mbc})
        return
        
