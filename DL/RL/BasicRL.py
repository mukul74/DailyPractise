import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import tensorboard

# Loading the environment for development
environment_name = 'CartPole-v0'
# env = gym.make(environment_name)

# episodes = 5
# for episode in range(1,episodes) :
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score = score + reward

#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()

log_path = 'C:\D_drive\Practise\DL\RL\Training\Logs'
model_save_path = os.path.join('C:\D_drive\Practise\DL\RL\Training\Saved_Models','PPO_Model_CartPoleRL')
# print(log_path)
env = gym.make(environment_name)
env = DummyVecEnv([lambda:env])
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
# model.learn(total_timesteps=20000)
# model.save(model_save_path)
# del model
model = PPO.load(model_save_path, env=env)
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

episodes = 5
for episode in range(1, episodes,1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward

    print('Episode : {} Score {}'.format(episode,score))


# training_log_path = "C:/D_drive/Practise/DL/RL/Training/Logs/PPO_2"
# tensorboard --logdir=(training_log_path)
