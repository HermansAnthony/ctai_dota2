import numpy as np
import cv2

import gym
from gym import spaces
from gym.utils import seeding

class ImageEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(255)
        img = cv2.imread('/home/jeroen/Desktop/test3.jpg',0) 
        xmax = img.shape[0]-1
        ymax = img.shape[1]-1
        low = np.array([0,0])
        high = np.array([xmax,ymax])
        print(img.shape)
        print(low, high)
        self.observation_space = spaces.Box(low, high, dtype=int)      
        self.iter = np.ndenumerate(img) 
        self._seed = self.seed
        self.new_img = np.zeros_like(img) 
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]   

    def _step(self, action):
        """
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        ob, real_action = next(self.iter, (None, None))
        if ob == None:
            cv2.imwrite('/home/jeroen/Desktop/test3_result.jpg',self.new_img)
            return np.array((0,0)), 255, True, {}
        self.new_img[ob[0]][ob[1]] = action
        reward = 255 - np.absolute(action - real_action)
        if reward < 240:
            reward = 0
        #if(ob[0] % 5 == 0 and ob[1] == 0):
        #    print(ob)
        #print(ob, reward)
        return ob, reward, False, {}

    def _reset(self):
        img = cv2.imread('/home/jeroen/Desktop/test3.jpg',0) 
        self.iter = np.ndenumerate(img) 
        return np.array((0,0))

    def _render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        pass

    def _get_reward(self):
        pass
