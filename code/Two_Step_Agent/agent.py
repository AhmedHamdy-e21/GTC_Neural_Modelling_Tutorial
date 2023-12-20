import numpy as np
class TwoStepAgent:

    def __init__(self, alpha1, alpha2, beta1, beta2, lam, w, p):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha1 -- learning rate for the first stage \in (0, 1]
            alpha2 -- learning rate for the second stage \in (0, 1]
            beta1  -- inverse temperature for the first stage
            beta2  -- inverse temperature for the second stage
            lam    -- eligibility trace parameter
            w      -- mixing weight for MF vs MB \in [0, 1] 
            p      -- perseveration strength
        '''

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1  = beta1
        self.beta2  = beta2
        self.lam    = lam
        self.w      = w
        self.p      = p
        


        return None
    
    def _init_counters(self):
        self.count_01_12 = 0
        self.count_02_11 = 0
        
    
    def _init_reward_probs(self):
        self.reward_probs = np.ones((2, 2)) * 0.5

        return None
        
    def _init_history(self):

        '''
        Initialise history to later compute stay probabilities
        '''

        self.history = np.empty((0, 3), dtype=int)

        return None
    
    def _update_history(self, a, s1, r1):

        '''
        Update history
        Input arguments:
            a  -- first stage action
            s1 -- second stage state
            r1 -- second stage reward
        '''

        self.history = np.vstack((self.history, [a, s1, r1]))

        return None

    def soft_max(self,q_values_net,rep,stage):
        beta = getattr( self, f'beta{stage}' )
        sum = np.sum( np.exp( beta * ( q_values_net + self.p * rep ) ) )
        ps = np.exp( beta * ( q_values_net + self.p * rep ) ) / sum
        
        return ps
    
    def update_reward(self):
        for i in range(2):
            for j in range(2):
                self.reward_probs[i, j] += np.random.normal(0, 0.025)

                if self.reward_probs[i, j] < 0.25:
                    self.reward_probs[i, j] = 0.5 - self.reward_probs[i, j]
                elif self.reward_probs[i, j] > 0.75:
                    self.reward_probs[i, j] = 1.5 - self.reward_probs[i, j]

        return self.reward_probs

    def get_stay_probabilities(self):

        '''
        Calculate stay probabilities
        '''

        common_r      = 0
        num_common_r  = 0
        common_nr     = 0
        num_common_nr = 0
        rare_r        = 0
        num_rare_r    = 0
        rare_nr       = 0
        num_rare_nr   = 0

        num_trials = self.history.shape[0]
        for idx_trial in range(num_trials-1):
            a, s1, r1 = self.history[idx_trial, :]
            a_next    = self.history[idx_trial+1, 0]

            # common
            if (a == 0 and s1 == 1) or (a == 1 and s1 == 2):
                # rewarded
                if r1 == 1:
                    if a == a_next:
                        common_r += 1
                    num_common_r += 1
                else:
                    if a == a_next:
                        common_nr += 1
                    num_common_nr += 1
            else:
                if r1 == 1:
                    if a == a_next:
                        rare_r += 1
                    num_rare_r += 1
                else:
                    if a == a_next:
                        rare_nr += 1
                    num_rare_nr += 1

        return np.array([common_r/num_common_r, rare_r/num_rare_r, common_nr/num_common_nr, rare_nr/num_rare_nr])


    def update_q_values_mb(self, q_values_td, id_stage_2, transition_probs):
        q_values_mb = np.zeros((2, 2))
        q_values_mb[0, 0] = transition_probs[0, 0] * np.max(q_values_td[1, :]) + transition_probs[0, 1] * np.max(q_values_td[2, :])
        q_values_mb[0, 1] = transition_probs[1, 0] * np.max(q_values_td[1, :]) + transition_probs[1, 1] * np.max(q_values_td[2, :])
        q_values_mb[1, :] = q_values_td[id_stage_2, :]
        
        return q_values_mb


        
    def simulate(self, num_trials):

        '''
        Main simulation function
        Input arguments:
            num_trials -- number of trials to simulate
        '''
        # Initialize agent history
        self._init_history()
        # Initialize counters
        self._init_counters()
        # Initialize reward probabilities
        self._init_reward_probs()

        # Create matrices for Q-values
        q_values_td = np.zeros((3, 2))
        q_values_mb = np.zeros((2, 2))

        # Create matrix for reward probabilities
        transition_probs = np.array([[0.7, 0.3], [0.3, 0.7]])
        
        
        # Initialize indicator function "vector"
        rep = np.zeros(2)

        # Run trials
        for i in range(num_trials):
            
            # First stage
            q_values_net = self.w * q_values_mb[0, :] + (1 - self.w) * q_values_td[0, :]
            ps_1 = self.soft_max(q_values_net,rep,1)
            action_1 = np.random.choice([0, 1], p=ps_1)
            
            is_common_state = False if np.random.rand() < 0.3 else True

            # Second stage
            id_stage_2 = 1 if (action_1 == 0 and is_common_state) or (action_1 == 1 and not is_common_state) else 2

            # Update transition learning counters
            if (action_1 == 0 and id_stage_2 == 1) or (action_1 == 1 and id_stage_2 == 2):
                self.count_01_12 += 1
            else:
                self.count_02_11 += 1

            # Update learned transition probabilities
            if self.count_01_12 >= self.count_02_11:
                transition_probs = np.array([[0.7, 0.3], [0.3, 0.7]])
                
            else:
                transition_probs = np.array([[0.3, 0.7], [0.7, 0.3]])


            # Calculate action probabilities for the second stage
            ps_2 = self.soft_max(q_values_td[id_stage_2, :],rep,2)
            # Take action
            action_2 = np.random.choice([0, 1], p=ps_2)
            
            # Calculate reward
            reward = 0
            if np.random.rand() <= self.reward_probs[id_stage_2 - 1, action_2]:
                reward = 1

            # Update Q-values for the first stage and second stage
            q_values_td[0, action_1] += self.alpha1 * (np.max(q_values_td[id_stage_2, :]) - q_values_td[0, action_1])
            q_values_td[id_stage_2, action_2] += self.alpha2 * (reward - q_values_td[id_stage_2, action_2])
            q_values_td[0, action_1] += self.alpha1 * self.lam * (reward - q_values_td[id_stage_2, action_2])

            # Update model-based Q-values
            q_values_mb = self.update_q_values_mb(q_values_td, id_stage_2, transition_probs)
    
            # Update agent's history
            self._update_history(action_1, id_stage_2, reward)

            # Update reward probabilities with Gaussian walk
            self.update_reward()
            # Update reputation vector (indicator function)
            rep = np.zeros(2)
            rep[action_1] = 1

        return None