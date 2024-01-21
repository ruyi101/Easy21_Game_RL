import random
import numpy as np

#################################################################################################
########################### Environment for Easy21 Game #########################################
#################################################################################################

class Environment:
    def __init__(self) -> None:
        self.dealerFirst = random.randint(1,10)
        self.playerSum = random.randint(1,10)
        self.accReward = 0
        self.reward = 0
        
    def restart(self):
        self.dealerFirst = random.randint(1,10)
        self.playerSum = random.randint(1,10)
        self.accReward = 0
        self.reward = 0
    
    def step(self, action):
        # As when one game ends, we set player's sum to False
        # Here, we check if the agent needs to restart the game.
        assert self.playerSum, "This game ends. Please restart!"
        
        ## action = 0 means hit
        ## action = 1 means stick
        if action==0:
            # if the action is hit
            # we draw a random number to decide the card number.
            number = random.randint(1,10)
            # we draw another random number to decide the color
            colorCoin = random.random()
            if colorCoin<=1/3:
                self.playerSum -= number
            else:
                self.playerSum += number
            
            
            # check if the game ends.      
            if self.playerSum<1 or self.playerSum > 21:
                    self.playerSum = False
                    self.accReward -= 1
                    self.reward = -1

        
        if action == 1:
            #If the action is stick. It is then the turn of the dealer.
            dealerSum = self.dealerFirst
            while dealerSum<17 and dealerSum>0:
                # If dealer's sum is smaller than 17, we draw a card
                dealerNum = random.randint(1,10)
                dealerColor = random.choice((-1,1, 1))
                dealerSum += dealerNum * dealerColor
                # Check if the game ends
                if dealerSum<1 or dealerSum>21:
                    self.playerSum = False
                    self.accReward += 1
                    self.reward = 1
                    return
                
            # If the dealer decide to stick.
            # Check who is the winner
            if dealerSum<self.playerSum:
                self.playerSum = False
                self.accReward += 1
                self.reward = 1
            elif dealerSum == self.playerSum:
                self.playerSum = False     
            else:
                self.playerSum = False
                self.accReward -= 1
                self.reward = -1

#################################################################################################
#################################################################################################
#################################################################################################


#################################################################################################
############################### Class Agent as the Base #########################################
#################################################################################################
class agent:
    def __init__(self, env) -> None:
        self.env = env
        self.Q = np.zeros((10, 21, 2))
    
    def get_random_action(self):
        return random.choice((0,1))
    
    def get_V(self):
        return np.max(self.Q, axis = 2)

#################################################################################################
#################################################################################################
#################################################################################################


#################################################################################################
################################ Monte Carlo Control ############################################
#################################################################################################
class MCagent(agent):
    def __init__(self, env, N0 = 100) -> None:
        super().__init__(env)
        self.N0 = N0
        self.N = np.zeros((10, 21))
        self.N_hist = np.zeros((10, 21, 2))
        
    
    
    def action(self):
        actionCoin = random.random()
        if actionCoin<= self.N0/(self.N0 + self.N[self.env.dealerFirst-1, self.env.playerSum-1]):
            return self.get_random_action()
        else:
            return np.argmax(self.Q[self.env.dealerFirst-1, self.env.playerSum-1])
    
    def refresh(self):
        self.N = np.zeros((10, 21))
        self.N_hist = np.zeros((10, 21, 2))
        
        
        
    
    def train(self, episode):
        self.refresh()
        
        for _ in range(episode):
            history = []
            self.env.restart()
            while self.env.playerSum:
                self.N[self.env.dealerFirst-1, self.env.playerSum - 1] += 1
                act = self.action()
                
                
                history.append((self.env.playerSum, act, self.env.accReward))
                
                self.env.step(act)
            history.append((self.env.playerSum, self.env.accReward))
            # print(history)
            
            
            for state, action, reward in history[:-1]:
                self.N_hist[self.env.dealerFirst-1, state-1, action] += 1
                self.Q[self.env.dealerFirst-1, state-1, action] += 1/(self.N_hist[self.env.dealerFirst-1, state-1, action]) * \
                                                                  (history[-1][1] - reward - self.Q[self.env.dealerFirst-1, state-1, action])
                
#################################################################################################
#################################################################################################
#################################################################################################


#################################################################################################
################################## TD control (Sarsa) ###########################################
#################################################################################################
class TDagent(agent):
    def __init__(self, env, N0 = 100, checkMSE = False) -> None:
        super().__init__(env)
        self.N0 = N0
        self.N = np.zeros((10, 21))
        self.N_hist = np.zeros((10, 21, 2))
        self.checkMSE = checkMSE
        
    
    
    def action(self):
        actionCoin = random.random()
        if actionCoin<= self.N0/(self.N0 + self.N[self.env.dealerFirst-1, self.env.playerSum-1]):
            return self.get_random_action()
        else:
            return np.argmax(self.Q[self.env.dealerFirst-1, self.env.playerSum-1])
    
    def refresh(self):
        self.N = np.zeros((10, 21))
        self.N_hist = np.zeros((10, 21, 2))
    
    def mse(self, optimal):
        return np.sum((self.Q - optimal)**2)
        
        
        
    
    def train(self, episodes, l, optimal = None):
        self.refresh()
        
        if self.checkMSE:
            self.MSE = np.zeros(episodes)
        
        for episode in range(episodes):
            E = np.zeros((10, 21, 2))
            
            self.env.restart()
            act = self.action()
            
            while self.env.playerSum:
                
                E[self.env.dealerFirst-1, self.env.playerSum-1, act] += 1
                self.N[self.env.dealerFirst-1, self.env.playerSum-1] += 1
                self.N_hist[self.env.dealerFirst-1, self.env.playerSum-1, act] += 1
                alpha = 1/self.N_hist[self.env.dealerFirst-1, self.env.playerSum-1, act]
                
                currSum = self.env.playerSum
                
                self.env.step(act)
                actNext = self.action()
                
                if self.env.playerSum:
                    delta = self.env.reward + self.Q[self.env.dealerFirst-1, self.env.playerSum-1, actNext] - \
                                self.Q[self.env.dealerFirst-1, currSum-1, act]
                else:
                    delta = self.env.reward -  self.Q[self.env.dealerFirst-1, currSum-1, act]

                self.Q += alpha*delta*E
                E *= l
                act = actNext
            
            if self.checkMSE:
                self.MSE[episode] = self.mse(optimal)
        
#################################################################################################
#################################################################################################
#################################################################################################


#################################################################################################
####################### Sarsa with Linear function Approximation ################################
#################################################################################################
def stateVec(dealer, player, action):
    dealer_features = [[1,4],[4,7],[7,10]]
    player_features = [[1,6], [4,9], [7,12], [10,15], [13,18], [16,21]]
    dealerFeature = np.array([x[0]<= dealer <=x[1] for x in dealer_features])
    playerFeature = np.array([x[0]<= player <=x[1] for x in player_features])
                    

    
    actionFeature = (1-action) * np.array([1,0]) + action * np.array([0,1])
    return np.kron(dealerFeature,np.kron(playerFeature,actionFeature))


class LFAagent():
    def __init__(self, env, epsilon = 0.05, alpha = 0.01, checkMSE = False) -> None:
        self.env = env
        self.w = np.array([random.gauss(0,1) for _ in range(36)])*0.1
        self.epsilon = epsilon
        self.alpha = alpha
        self.checkMSE = checkMSE
        
    
    def action(self):
        actionCoin = random.random()
        if actionCoin <= self.epsilon:
            return random.choice((0,1))
        else:
            return np.argmax([self.get_Q(0), self.get_Q(1)])
    
    def get_Q(self, action):
        return np.inner(self.w, stateVec(self.env.dealerFirst, self.env.playerSum, action))
    
    def Q_table(self):
        Q = np.zeros((10,21,2))
        for dealer in range(1, 11):
            for player in range(1, 22):
                for action in [0,1]:
                    Q[dealer-1, player-1, action] = np.inner(self.w, stateVec(dealer, player, action))
        return Q
    
    def get_V(self):
        return np.max(self.Q_table(), axis = 2)
    
    def mse(self, optimal):
        return np.sum((self.Q_table() - optimal)**2)
            
    
    def train(self, numEpisode, l, optimal = None):
        
        if self.checkMSE:
            self.MSE = np.zeros(numEpisode)
        
        for episode in range(numEpisode):
            self.env.restart()
            act = self.action()
            E = np.zeros(36)
            
            while self.env.playerSum:
                E += stateVec(self.env.dealerFirst, self.env.playerSum, act)
                Q_prev = self.get_Q(act)
                self.env.step(act)
                
                actNext = self.action()
                
                if self.env.playerSum:
                    delta = self.env.reward + self.get_Q(actNext) - Q_prev
                else:
                    delta = self.env.reward -  Q_prev
                
                self.w += self.alpha * delta * E
                
                E *= l
                act = actNext
                
            if self.checkMSE:
                self.MSE[episode] = self.mse(optimal)
                
#################################################################################################
#################################################################################################
#################################################################################################

