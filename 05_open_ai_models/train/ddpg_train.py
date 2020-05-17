# import required libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

def ddpg_train(env, agent, signal, n_episodes=10000, print_every=100):
	
	# set up the deque list
	scores_deque = deque(maxlen=print_every)

	# initialise the empty scores list
	scores = []

	# run through the episodes
	for i_episode in range(1, n_episodes+1):
		
		# initialise the environment, agent, demand signal, and score
		state = env.reset()
		agent.reset()
		signal.reset(1 , 'on', 0.01 , 'off', 0.0)
		
		out_s_1 = [0]
		out_s_2 = [0]
		control_s_1 = [0]
		control_s_2 = [0]
		time = [0]

		score = 0

		# run through the episode
		while True:
			
			# take control action
			action = agent.act(state)

			out_s_1.append(state[2])
			out_s_2.append(state[6])
			time.append(env.t)
			control_s_1.append(action[0])
			control_s_2.append(action[1])

			# determine demand
			demand = (signal.del_p_L_1_func(env.t),
					  signal.del_p_L_2_func(env.t))
			
			# step environment forward in time
			next_state, reward, done, _ = env.step(action, demand)

			# step agent forward in time
			agent.step(state, action, reward, next_state, done)

			# save the future state as the current state
			state = next_state

			# save the reward from the time step
			score += reward

			# if episode is completed finish
			if done:
				break

		# save scores to deque and total
		scores_deque.append(score)
		scores.append(score)

		# print updates to the terminal
		print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
		if i_episode % print_every == 0:
			print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

		# save agent progress
		torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
		torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

		if i_episode % 50 == 0:
			
			plt.subplot(311)
			plt.plot(time, out_s_1)
			plt.plot(time, out_s_2)

			plt.subplot(312)
			plt.plot(time, control_s_1)

			plt.subplot(313)
			plt.plot(time, control_s_2)

			plt.savefig('plot_{}.png'.format(i_episode))
			plt.clf()

		if np.mean(scores_deque) > 20.0:
			break

	return scores