import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import pickle


def capture_agent_progress(time_list,					# time
                           out_s_1, out_s_2,			# frequency signals
                           control_s_1, control_s_2,	# control signals
                           out_tieline,					# tieline signal
                           demand_list,					# power demand
                           png_plot_file_path,			# file path
                           pik_file_path):				# file path

    # Plot frequency responses
    plt.subplot(511)
    plt.plot(time_list, out_s_1)
    plt.plot(time_list, out_s_2)

    # Plot control signal 1
    plt.subplot(512)
    plt.plot(time_list, out_tieline)

    # Plot control signal 1
    plt.subplot(513)
    plt.plot(time_list, control_s_1)

    # Plot control signal 2
    plt.subplot(514)
    plt.plot(time_list, control_s_2)

    # Plot demand signal
    plt.subplot(515)
    plt.plot(time_list, demand_list)

    plt.savefig(png_plot_file_path)
    plt.clf()

    # Save data in serialised format
    pik_list = [time_list,
                out_s_1, out_s_2,
                out_tieline,
                control_s_1, control_s_2,
                demand_list]

    with open(pik_file_path, 'wb') as f:
        pickle.dump(pik_list, f)


def capture_agent_score_progress(scores,
                                 png_plot_file_path,
                                 pik_file_path):

    # Plot the rewards vs episode
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(png_plot_file_path)
    plt.clf()

    # Save data in serialised format
    with open(pik_file_path, 'wb') as f:
        pickle.dump(scores, f)