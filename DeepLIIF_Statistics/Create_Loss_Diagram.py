import collections
import numpy as np
import matplotlib.pyplot as plt

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def read_losses(file_name):
    losses = {}
    with open(file_name) as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('(epoch'):
            line = line.replace(',', '').replace('(', '').replace(')', '').replace(':', '').strip()
            values = line.split(' ')
            current_losses = {}
            epoch_number = -1
            for i in range(len(values)):
                if values[i] == 'epoch':
                    epoch_number = int(values[i + 1])
                else:
                    if not isfloat(values[i]) and values[i] != 'time' and values[i] != 'iters' and values[i] != 'data':
                        print(values[i], values[i + 1])
                        current_losses[values[i]] = float(values[i + 1])

            losses[epoch_number] = current_losses
    loss_values = collections.defaultdict(list)
    for key in losses:
        loss_values['epoch'].append(key)
        for k in losses[key].keys():
            loss_values[k].append(losses[key][k])

    return loss_values


def create_loss_diagram(file_name):
    loss_values = read_losses(file_name)
    x = loss_values['epoch']
    plt.figure(figsize=(12, 4))
    for i in range(1, 6):
        plt.plot(x, loss_values['G_GAN_' + str(i)], label='G_GAN_' + str(i))
    # plt.legend()
    # plt.show()

    # plt.figure()
    for i in range(1, 6):
        plt.plot(x, loss_values['G_L1_' + str(i)], label='G_L1_' + str(i))
    # plt.legend()
    # plt.show()

    # plt.figure()
    for i in range(1, 6):
        plt.plot(x, loss_values['D_real_' + str(i)], label='D_real_' + str(i))
    # plt.legend()
    # plt.show()

    # plt.figure()
    for i in range(1, 6):
        plt.plot(x, loss_values['D_fake_' + str(i)], label='D_fake_' + str(i))
    # plt.legend()
    # plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=5)
    plt.legend(ncol=5)
    plt.show()
    # plt.scatter(x, y)
    # plt.plot(x, y)
    # plt.title("Connected Scatterplot points with line")
    # plt.xlabel("epoch")
    # plt.ylabel("G_GAN_1")
    # plt.show()
    # figure.tight_layout()
    # print(loss_values)


# create_loss_diagram('D://DeepLIIF//checkpoints//DeepLIIF_Empty_500_Model//loss_log.txt')
# create_loss_diagram('C://Users//localadmin//Desktop//loss_log_SpectralNorm_SYN.txt')
create_loss_diagram('C://Users//localadmin//Desktop//loss_log_DeepLIIF.txt')