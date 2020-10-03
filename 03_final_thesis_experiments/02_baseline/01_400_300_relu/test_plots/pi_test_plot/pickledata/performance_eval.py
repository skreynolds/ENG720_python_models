# import required libraries
import pandas
import numpy as np
import pickle


# max calculation
def max_mag(sig):
    val = max(abs(sig))
    return val


# avg calculation
def avg_mag(sig):
    val = np.mean(abs(sig))
    return val


# settling time calculation
def settling_time(sig,t):
    val = t[next(len(sig)-i for i in range(2,len(sig)-1) if abs(sig[-i]- sig[-1])>0.001)] - t[0]
    return val


# performance evaluation
def performance_evaluation(pik):

    t = np.array(pik[0])
    sig_1 = np.array(pik[1])
    sig_2 = np.array(pik[2])
    ctl_1 = np.array(pik[4])
    ctl_2 = np.array(pik[5])

    ########################################
    # Area 1
    ########################################

    # area 1 max freq
    max_freq_1 = max_mag(sig_1)

    # area 1 average freq
    avg_freq_1 = avg_mag(sig_1)

    # area 1 max control effort
    max_ctl_1 = max_mag(ctl_1)

    # area 1 average control effort
    avg_ctl_1 = avg_mag(ctl_1)

    # area 1 settling time
    ts_1 = settling_time(sig_1, t)

    # Print results for Area 1
    print('Max freq dev Area 1: \t\t {}'.format(max_freq_1))
    print('Avg freq dev Area 1: \t\t {}'.format(avg_freq_1))
    print('Max ctl effort Area 1: \t\t {}'.format(max_ctl_1))
    print('Avg ctl effort Area 1: \t\t {}'.format(avg_ctl_1))
    print('Settling time: \t\t\t {}'.format(ts_1))

    ########################################
    # Area 2
    ########################################

    # area 2 max freq
    max_freq_2 = max_mag(sig_2)

    # area 2 average freq
    avg_freq_2 = avg_mag(sig_2)

    # area 2 max control effort
    max_ctl_2 = max_mag(ctl_2)

    # area 2 average control effort
    avg_ctl_2 = avg_mag(ctl_2)

    # area 2 settling time
    ts_2 = settling_time(sig_2, t)

    # Print results for Area 1
    print('Max freq dev Area 2: \t\t {}'.format(max_freq_2))
    print('Avg freq dev Area 2: \t\t {}'.format(avg_freq_2))
    print('Max ctl effort Area 2: \t\t {}'.format(max_ctl_2))
    print('Avg ctl effort Area 2: \t\t {}'.format(avg_ctl_2))
    print('Settling time: \t\t\t {}'.format(ts_2))


# main function
def main():
    '''
    magnitude = ['neg', 'pos']
    duration = [5, 10, 15, 20, 25]

    for mag in magnitude:
        for dur in duration:

            with open('{}_{}_plot_final.pkl'.format(mag,dur), 'rb') as f:
                ddpg_results = pickle.load(f)

            print('Performance evaluation for {} {}'.format(mag,dur))
            performance_evaluation(ddpg_results)
    '''
    with open('zz_plot_final.pkl', 'rb') as f:
                ddpg_results = pickle.load(f)

    print('Performance evaluation for PID')
    performance_evaluation(ddpg_results)
    '''
    df = pandas.DataFrame(ddpg_results).T
    df.to_csv("verification.csv", sep=',', index=False)
    '''



# call to main from terminal
if __name__ == '__main__':
    main()
