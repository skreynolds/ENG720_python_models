# import required libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from collections import deque

# import utility functions
from utils import *

def ddpg_train(env, agent, signal, n_episodes=2000, print_every=100):

	######################################################################
	# Initialisation
	######################################################################
	# set up the deque list
	scores_deque = deque(maxlen=print_every)

	# initialise the empty scores list
	scores = []

	# highest score
	highest_score = None

	######################################################################
	# Train agent (i.e. run through all episodes)
	######################################################################
	for i_episode in range(1, n_episodes+1):

		# initialise the environment, agent, demand signal, and score
		state = env.reset()
		agent.reset()
		signal.reset(1 , 'on', 0.01 , 'off', 0.0)

		# initialise storage for progress reporting
		out_s_1 = [0]
		out_s_2 = [0]
		out_tieline = [0]
		control_s_1 = [0]
		control_s_2 = [0]
		demand_list = [0]
		time_list = [0]

		score = 0

		##############################################
		# Run through single episode
		##############################################
		while True:

			# take control action
			action = agent.act(state)

			# determine demand
			demand = (signal.del_p_L_1_func(env.t),
					  signal.del_p_L_2_func(env.t))

			# capture data for progress reporting
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

		######################################################################
		# Print updates to the terminal
		######################################################################
		print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
		if i_episode % print_every == 0:
			print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

		######################################################################
		# Period printing of figures, pickle data, and saving agent progress
		######################################################################
		if i_episode % 50 == 0:

			# Plot the agent performance png files
			png_plot_file_path = './progress_plots/periodic_plot/pngplot/plot_{}.png'.format(i_episode)
			pik_file_path = './progress_plots/periodic_plot/pickledata/plot_{}.pkl'.format(i_episode)
			capture_agent_progress(time_list,
								   out_s_1, out_s_2,
								   control_s_1, control_s_2
								   out_tieline,
								   demand_list,
								   png_plot_file_path,
								   pik_file_path)

			# Plot the reward vs episode
			png_plot_file_path = './progress_plots/periodic_reward/pngplot/plot_{}.png'.format(i_episode)
			pik_file_path = './progress_plots/periodic_reward/pickledata/plot_{}.pkl'.format(i_episode)
			captures_agent_score_progress(scores, png_plot_file_path, pik_file_path)

			# save agent progress
			actor_file_path = './saved_agents/periodic_agent_save/checkpoint_actor_{}.pth'.format(i_episode)
			critic_file_path = './saved_agents/periodic_agent_save/checkpoint_critic_{}.pth'.format(i_episode)
			torch.save(agent.actor_local.state_dict(), actor_file_path)
			torch.save(agent.critic_local.state_dict(), critic_file_path)

		######################################################################
		# Best Agents figures, pickle data, and saving agent progress
		######################################################################
		if (highest_score == None) or (np.mean(scores_deque) > highest_score):

			# Update highest scores
			highest_score = np.mean(scores_deque)

			# Plot the png files
			png_plot_file_path = './progress_plots/highest_average_score_plot/pngplot/plot_{}.png'.format(i_episode)
			pik_file_path = './progress_plots/highest_average_score_plot/pickledata/plot_{}.pkl'.format(i_episode)
			capture_agent_progress(time_list,
								   out_s_1, out_s_2,
								   control_s_1, control_s_2
								   out_tieline,
								   demand_list,
								   png_plot_file_path,
								   pik_file_path)

			# Plot the reward vs episode
			png_plot_file_path = './progress_plots/highest_average_score_reward/pngplot/plot_{}.png'.format(i_episode)
			pik_file_path = './progress_plots/highest_average_score_reward/pickledata/plot_{}.pkl'.format(i_episode)
			captures_agent_score_progress(scores, png_plot_file_path, pik_file_path)

			# save agent progress
			actor_file_path = './saved_agents/highest_average_score/checkpoint_actor_{}.pth'.format(i_episode)
			critic_file_path = './saved_agents/highest_average_score/checkpoint_critic_{}.pth'.format(i_episode)
			torch.save(agent.actor_local.state_dict(), actor_file_path)
			torch.save(agent.critic_local.state_dict(), critic_file_path)

		####################################################
		# Uncomment if early termination is required
		####################################################
		#if np.mean(scores_deque) > 2000.0:
		#	break


	####################################################
	# Final capture
	####################################################
	# Plot the agent performance png files
	png_plot_file_path = './progress_plots/periodic_plot/pngplot/zz_plot_final.png'
	pik_file_path = './progress_plots/periodic_plot/pickledata/zz_plot_final.pkl'
	capture_agent_progress(time_list,
						   out_s_1, out_s_2,
						   control_s_1, control_s_2
						   out_tieline,
						   demand_list,
						   png_plot_file_path,
						   pik_file_path)

	# Plot the reward vs episode
	png_plot_file_path = './progress_plots/periodic_plot/pngplot/zz_plot_final.png'
	pik_file_path = './progress_plots/periodic_plot/pickledata/zz_plot_final.pkl'
	captures_agent_score_progress(scores, png_plot_file_path, pik_file_path)

	# save agent progress
	actor_file_path = './saved_agents/periodic_agent_save/zz_checkpoint_actor_final.pth'
	critic_file_path = './saved_agents/periodic_agent_save/zz_checkpoint_critic_final.pth'
	torch.save(agent.actor_local.state_dict(), actor_file_path)
	torch.save(agent.critic_local.state_dict(), critic_file_path)

	return scores