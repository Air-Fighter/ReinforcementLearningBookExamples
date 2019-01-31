import numpy as np
from collections import defaultdict
import sys
import matplotlib.pyplot as plt

class CliffWalking(object):
    def __init__(self, step_limit=50, birth=[3,0]):
        self.shape = (4, 12)
        self.birth = birth
        self.pos = tuple(birth)
        self.cliff = np.zeros(self.shape, dtype=np.bool)
        self.cliff[-1, 1:-1] = 1
        self.done = False  # Show whether the game is over
        self.num_action = 4
        self.actions = {
            'U': 0,
            'D': 1,
            'L': 2,
            'R': 3
        }

        self.action2shift = {
            0: [-1, 0],
            1: [1, 0],
            2: [0, -1],
            3: [0, 1]
        }

    def take_action(self, action):
        self.done = False
        reward = -1
        pos_0,pos_1 = self.pos[0],self.pos[1]
        if isinstance(action,str):
            action = self.actions[action]
        pos_0 += self.action2shift[action][0]
        pos_1 += self.action2shift[action][1]
        if pos_0<0 or pos_0>self.shape[0]-1 or pos_1<0 or pos_1>self.shape[1]-1:
            reward = -1
            pos_0 = max(pos_0, 0)
            pos_0 = min(pos_0, self.shape[0] - 1)
            pos_1 = max(pos_1, 0)
            pos_1 = min(pos_1, self.shape[1] - 1)
        
        self.pos = (pos_0,pos_1)
        
        if self.cliff[self.pos[0]][self.pos[1]]:
            reward = -100.
            self.reset()
        elif self.pos[0] == self.shape[0] - 1 and self.pos[1] == self.shape[1] - 1:
            self.done = True
            reward = 0
        else: pass

        return self.pos, reward, self.done

    def show_pos(self):
        env = np.zeros(self.shape)
        env[self.pos[0]][self.pos[1]] = 1
        print(env)
    
    def show_path(self,states):
        env = np.zeros(self.shape)
        cnt = 0
        for s in states:
            cnt += 1
            env[s[0]][s[1]]=cnt
        print(env)
    
    def get_pos(self):
        return self.pos
    
    def reset(self):
        self.pos = tuple(self.birth)
        self.done = False
        return self.pos, self.done
    
def test_cliff_warlking_by_hand(env):
    _,terminate = env.reset()
    env.show_pos()
    score = 0
    while not terminate:
        action = input('input your actoin (U/D/L/R):')
        new_pos, reward, terminate = env.take_action(action)
        score += reward
        print(new_pos, reward, terminate, score)
        env.show_pos()    
  
    
def epsilon_greedy_policy(Q,state,num_action,epsilon=0.1,greedy=False):
    '''
    greedy_action = np.argmax(Q[state])
    if np.random.uniform(0,1)>epsilon or greedy == True:      # Greedy
        return greedy_action
    else:
        return np.random.randint(0,num_action-1)
    '''
    best_action = np.argmax(Q[state])
    A =np.ones(num_action,dtype=np.float32)*epsilon/num_action
    A[best_action] += 1-epsilon
    if greedy==False:
        return np.random.choice(np.arange(num_action),p=A)
    else:
        return best_action


def test_cliff(Q,max_step=300):
    steps = 0
    env_test = CliffWalking()
    state, terminate = env_test.reset()
    action_store = []
    state_store = []
    sum_reward = 0
    while not terminate:
        steps += 1
        action = epsilon_greedy_policy(Q,state,env.num_action,greedy=False)
        state, reward, terminate = env_test.take_action(action)
        sum_reward += reward
        action_store.append(action)
        state_store.append(state)
        if steps > max_step:
            print('Out of steps')
            return state_store, sum_reward, action_store
    return state_store, sum_reward, action_store

        

def sarsa(env,episode_nums,discount_factor=1.0, alpha=0.5,Q=None):
    #env = CliffWalking()
    if Q==None:
        Q = defaultdict(lambda:np.zeros(env.num_action))
    rewards=[]
    epsilon = 0.1
    for i_episode in range(1,1+episode_nums):
        state, terminate = env.reset()
        action = epsilon_greedy_policy(Q,state,env.num_action,epsilon=epsilon,greedy=False)
        sum_reward = 0
        while not terminate:
            new_state,reward,terminate = env.take_action(action)
            if terminate:
                Q[state][action]=Q[state][action]+alpha*(reward+discount_factor*0.0-Q[state][action])
                break
            else:
                new_action =  epsilon_greedy_policy(Q,new_state,env.num_action,epsilon=epsilon,greedy=False)
                Q[state][action]=Q[state][action]+alpha*(reward+discount_factor*Q[new_state][new_action]-Q[state][action])
                state = new_state
                action = new_action
            sum_reward += reward
        rewards.append(sum_reward)
        
        if i_episode % 100 == 5:
            #epsilon = epsilon/2
            print("\rEpisode {}/{}. epsilon={}".format(i_episode, episode_nums,epsilon))
            sys.stdout.flush()
            states,test_reward,actions = test_cliff(Q)
            if test_reward < -300:
                print(states)
                print(Q[states[-1]])
            rewards.append(test_reward)
            
    return Q,rewards

env = CliffWalking()
episode_nums=10000
Q,rewards = sarsa(env,episode_nums)



states,reward,actions = test_cliff(Q)
env.show_path(states)



