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
		out_tieline = [0]
		control_s_1 = [0]
		control_s_2 = [0]
		demand_list = [0]
		time_list = [0]

		score = 0

		# run through the episode
		while True:
			
			# take control action
			action = agent.act(state)

			# determine demand
			demand = (signal.del_p_L_1_func(env.t),
					  signal.del_p_L_2_func(env.t))

			out_s_1.append(state[2])
			out_s_2.append(state[6])
			out_tieline.append(state[3])
			control_s_1.append(action[0])
			control_s_2.append(action[1])
			demand_list.append(demand[0])
			time_list.append(env.t)
			
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
			
			plt.subplot(511)
			plt.plot(time_list, out_s_1)
			plt.plot(time_list, out_s_2)

			plt.subplot(512)
			plt.plot(time_list, out_tieline)

			plt.subplot(513)
			plt.plot(time_list, control_s_1)

			plt.subplot(514)
			plt.plot(time_list, control_s_2)

			plt.subplot(515)
			plt.plot(time_list, demand_list)

			plt.savefig('plot_{}.png'.format(i_episode))
			plt.clf()

		if np.mean(scores_deque) > -100.0:
			break

	return scores