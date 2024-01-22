
# 11/01/23
# 11a[cc] / Alejandro Gutierrez Acosta

import numpy as np
import matplotlib.pyplot as plt

# Showcase the change in the line slope and intercept as the
# perceptron learning algorithm calculates the target function
# Each line is supposed to have a more or less distinct colour
# brightness from the previous one

def hexhex(num_iter):
    # Produce a reasonable hex code for each line step given
    # @returns the list of lines with the hex code cut into each list item
    target_r, target_g, target_b = 128, 255, 128

    r, g, b = 0, 0, 0
    r_increment = max(1, (target_r - r) // num_iter)
    g_increment = max(1, (target_g - g) // num_iter)
    b_increment = max(1, (target_b - b) // num_iter)

    hex_codes = []

    for step in range(num_iter):
        r = min(r + r_increment, target_r)
        g = min(g + g_increment, target_g)
        b = min(b + b_increment, target_b)

        hc = "#{:02X}{:02X}{:02X}".format(r, g, b)
        hex_codes.append(hc)
        #print(f"hexs: {hex_codes}")

    #print(f"hexs length: {len(hex_codes)}")

    return hex_codes


def pla(dataset_size):
    # Run the PLA for a random line and given dataset_size
    # of a bidimensional dataset
    # @return output from hexhex() function by providing line array

    f_x = np.array([-1.0, 1.0])
    f_m = np.random.uniform(-3.5, 3.5)
    f_b = np.random.uniform(-0.5, 0.5)
    f_y = f_m * f_x + f_b

    # dataset
    biDatasetD = np.random.uniform(-1.0, 1.0, (dataset_size, 2))
    biDatasetD = np.c_[np.ones(biDatasetD.shape[0]), biDatasetD]
    labels = np.sign(f_m * biDatasetD[:, 1] - biDatasetD[:, 2] + f_b)
    weights = np.zeros(2 + 1)

    # save each line iteration
    step_y = []

    num_iter = 0
    while True:
        num_iter += 1
        misclassified = 0
        for i, point in enumerate(biDatasetD):
            prediction = np.sign(np.dot(weights, point))
            if prediction != labels[i]:
                weights = weights + labels[i] * point
                misclassified += 1

        a_m = (-weights[1]) / weights[2]
        a_b = weights[0] / weights[2]
        step_y.append(a_m * f_x - a_b)

        if misclassified == 0:
            break

    pla_x = np.array([-1.0, 1.0])
    pla_m = (-weights[1]) / weights[2]
    pla_b = weights[0] / weights[2]
    pla_y = pla_m * pla_x - pla_b

    print(f"dataset size: {dataset_size}")
    print(f"iterations: {num_iter}")
    #print(f"steps length: {len(step_y)}")

    return step_y, hexhex(num_iter), pla_y, f_y, biDatasetD, labels, num_iter


def main():
    # random set seed for testing
    #np.random.seed(311)
    dataset_size = 250

    step_y, hex_codes, pla_y, f_y, biDatasetD, labels, num_iter = pla(dataset_size)

    # plotting
    plt.figure(figsize=(12, 8))
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)

    plt.scatter(biDatasetD[:, 1], biDatasetD[:, 2], c=labels)

    pla_x = np.array([-1.0, 1.0])
    plt.plot(pla_x, f_y, label="target", c="red", zorder=99)
    plt.plot(pla_x, pla_y, label="final", c="#FF00FF", zorder=98)
    plt.plot(pla_x, pla_x, label="starting", c="black", zorder=97)
    plt.legend()

    for i in range(num_iter):
        plt.plot(pla_x, step_y[i], c=hex_codes[i])
        plt.pause(0.25)

    plt.show(block=True)


if __name__ == "__main__":
    main()