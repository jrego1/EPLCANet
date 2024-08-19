import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")


import os
import random
from datetime import datetime
from lcapt.lca import LCAConv2D
from lcapt.analysis import make_feature_grid
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "elapsed time : %s \t (will finish in %s)" % (asMinutes(s), asMinutes(rs))


def get_estimate(dic):
    estimates = {}
    for key in dic.keys():
        if key.find("weight") != -1:
            estimate = integrate(dic[key])
            estimates[key] = estimate[-1, :]
    return estimates


def half_sum(dic1, dic2):
    dic3 = {}
    for key in dic1.keys():
        dic3[key] = (dic1[key] + dic2[key]) / 2
    return dic3


def compare_estimate(bptt, ep_1, ep_2, path):
    # Compare the estimate of the BPTT with two different estimates from EP
    heights = []
    abscisse = []
    plt.figure(figsize=(16, 9))
    for key in bptt.keys():

        ep_3 = (ep_1[key] + ep_2[key]) / 2

        ep1_bptt = (ep_1[key] - bptt[key]).abs()
        ep2_bptt = (ep_2[key] - bptt[key]).abs()
        ep3_bptt = (ep_3 - bptt[key]).abs()

        comp = torch.where(
            (ep1_bptt + ep2_bptt) == 0,
            torch.ones_like(ep1_bptt),
            (2 * ep3_bptt) / (ep1_bptt + ep2_bptt),
        )
        comp = comp.mean().item()

        if key.find("weight") != -1:
            heights.append(comp)
            abscisse.append(int(key[9]) + 1)

    plt.bar(abscisse, heights)
    plt.ylim((0.0, 1.0))
    plt.title(
        "Euclidian distance between EP symmetric and BPTT, divided by mean distance between EP one-sided and BPTT\n 1.0 means EP symmetric is as close to BPTT as EP one-sided, 0.5 means EP symmetric twice closer to BPTT than EP one-sided"
    )
    plt.ylabel("Relative distance to BPTT")
    plt.xlabel("Layer index")
    plt.savefig(path + "/bars.png", dpi=300)
    plt.close()


def integrate(x):
    y = torch.empty_like(x)
    with torch.no_grad():
        for j in reversed(range(x.shape[0])):
            integ = 0.0
            for i in range(j):
                integ += x[i]
            y[j] = integ
    return y


def plot_gdu(BPTT, EP, path, EP_2=None, alg="EP"):
    # Plots gradient updates of BPTT and EP.
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for key in EP.keys():
        fig = plt.figure(figsize=(16, 9))
        for idx in range(3):
            if len(EP[key].size()) == 3:
                i, j = np.random.randint(EP[key].size(1)), np.random.randint(
                    EP[key].size(2)
                )
                ep = EP[key][:, i, j].cpu().detach()
                if EP_2 is not None:
                    ep_2 = EP_2[key][:, i, j].cpu().detach()
                bptt = BPTT[key][:, i, j].cpu().detach()
            elif len(EP[key].size()) == 2:
                i = np.random.randint(EP[key].size(1))
                ep = EP[key][:, i].cpu().detach()
                if EP_2 is not None:
                    ep_2 = EP_2[key][:, i].cpu().detach()
                bptt = BPTT[key][:, i].cpu().detach()
            elif len(EP[key].size()) == 5:
                i, j = np.random.randint(EP[key].size(1)), np.random.randint(
                    EP[key].size(2)
                )
                k, l = np.random.randint(EP[key].size(3)), np.random.randint(
                    EP[key].size(4)
                )
                ep = EP[key][:, i, j, k, l].cpu().detach()
                if EP_2 is not None:
                    ep_2 = EP_2[key][:, i, j, k, l].cpu().detach()
                bptt = BPTT[key][:, i, j, k, l].cpu().detach()
            ep, bptt = integrate(ep), integrate(bptt)
            ep, bptt = ep.numpy().flatten(), bptt.numpy().flatten()
            plt.plot(
                ep,
                linestyle=":",
                linewidth=2,
                color=colors[idx],
                alpha=0.7,
                label=alg + " one-sided right",
            )
            plt.plot(bptt, color=colors[idx], linewidth=2, alpha=0.7, label="BPTT")
            if EP_2 is not None:
                ep_2 = integrate(ep_2)
                ep_2 = ep_2.numpy().flatten()
                plt.plot(
                    ep_2,
                    linestyle=":",
                    linewidth=2,
                    color=colors[idx],
                    alpha=0.7,
                    label=alg + " one-sided left",
                )
                plt.plot(
                    (ep + ep_2) / 2,
                    linestyle="--",
                    linewidth=2,
                    color=colors[idx],
                    alpha=0.7,
                    label=alg + " symmetric",
                )
            plt.title(key.replace(".", " "))
        plt.grid()
        plt.legend()
        plt.xlabel("time step t")
        plt.ylabel("gradient estimate")
        fig.savefig(path + "/" + key.replace(".", "_") + ".png", dpi=300)
        plt.close()


def plot_neural_activity(neurons, path):
    # Plot histogram of the activity of each neuron in each layer.
    N = len(neurons)
    fig = plt.figure(figsize=(3 * N, 6))
    for idx in range(N):
        fig.add_subplot(2, N // 2 + 1, idx + 1)
        nrn = neurons[idx].cpu().detach().numpy().flatten()
        plt.hist(nrn, 50)
        plt.title("neurons of layer " + str(idx + 1))
    fig.savefig(path + "/neural_activity.png")
    plt.tight_layout()
    plt.close()


def plot_synapses(model, path):
    # Plot the distribution of synapses in each layer of the given model.
    if isinstance(model, nn.DataParallel):
        N = len(model.module.synapses)
    else:
        N = len(model.synapses)
    fig = plt.figure(figsize=(4 * N, 3))
    for idx in range(N):
        fig.add_subplot(1, N, idx + 1)
        nrn = model.synapses[idx].weight.cpu().detach().numpy().flatten()
        plt.hist(nrn, 50)
        plt.title("synapses of layer " + str(idx + 1))
    fig.savefig(path)
    plt.close()


def plot_lca_weights(model, path):
    if isinstance(model.synapses, nn.DataParallel):
        N = len(model.module.synapses)
    else:
        N = len(model.synapses)
    for idx in range(N):
        if isinstance(model.synapses[idx], LCAConv2D):
            weights = make_feature_grid(model.synapses[idx].get_weights())
            plt.imshow(weights.float().cpu().numpy())
            plt.title("LCANet Dictionary")
            plt.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            plt.tight_layout()
            plt.savefig(path)


def plot_lca_acts(acts, threshold=0.1):
    """
    Plot a histogram of the number of images each dictionary element in the LCA layer is active.

    Parameters:
    - lca_activations (np.array): Array of LCA activations with shape (num_images, num_dictionary_elements).
    - threshold (float): Threshold to determine if a dictionary element is active.
    """
    # Convert activations to binary (active/inactive) based on the threshold
    binary_activations = (acts > threshold).astype(int)

    # Sum across images to get the count of active images for each dictionary element
    active_counts = np.sum(binary_activations, axis=0)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(
        active_counts,
        bins=range(active_counts.max() + 1),
        edgecolor="black",
        align="left",
    )
    plt.xlabel("Number of Images Active")
    plt.ylabel("Number of Dictionary Elements")
    plt.title("Dictionary Element Activations in LCA Layer")
    plt.grid(True)
    plt.show()


def plot_lines(line_1, line_2, title, label_1, label_2, x_label, y_label, path):
    fig = plt.figure(figsize=(16, 9))
    x_axis = [i for i in range(len(line_1))]
    plt.plot(x_axis, line_1, label=label_1)
    plt.plot(x_axis, line_2, label=label_2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    fig.savefig(path)
    plt.tight_layout()
    plt.close()


def createHyperparametersFile(path, args, model, command_line):

    hyperparameters = open(path + r"/hyperparameters.txt", "w+")
    L = [
        "- task: {}".format(args.task) + "\n",
        "- data augmentation (if CIFAR10): {}".format(args.data_aug) + "\n",
        "- learning rate decay: {}".format(args.lr_decay) + "\n",
        "- scale for weight init: {}".format(args.scale) + "\n",
        "- activation: {}".format(args.act) + "\n",
        "- learning rates: {}".format(args.lrs) + "\n",
        "- weight decays: {}".format(args.wds) + "\n",
        "- momentum (if sgd): {}".format(args.mmt) + "\n",
        "- optimizer: {}".format(args.optim) + "\n",
        "- loss: {}".format(args.loss) + "\n",
        "- alg: {}".format(args.alg) + "\n",
        "- minibatch size: {}".format(args.mbs) + "\n",
        "- T1: {}".format(args.T1) + "\n",
        "- T2: {}".format(args.T2) + "\n",
        "- betas: {}".format(args.betas) + "\n",
        "- random beta_2 sign: {}".format(args.random_sign) + "\n",
        "- thirdphase: {}".format(args.thirdphase) + "\n",
        "- softmax: {}".format(args.softmax) + "\n",
        "- same update VFCNN: {}".format(args.same_update) + "\n",
        "- epochs: {}".format(args.epochs) + "\n",
        "- seed: {}".format(args.seed) + "\n",
        "- device: {}".format(args.device) + "\n",
    ]

    print(command_line, "\n", file=hyperparameters)
    hyperparameters.writelines(L)
    if hasattr(model, "pools"):
        print("\nPoolings :", model.pools, "\n", file=hyperparameters)
    else:
        print("\n")
    print(model, file=hyperparameters)

    hyperparameters.close()


def read_hyperparameters(file_path: str) -> dict:
    with open(file_path, "r") as file:
        lines = file.readlines()

    load_params = {}
    for line in lines:
        if line.startswith("-"):
            continue
        else:
            parts = line.split()
            i = 0
            while i < len(parts):
                if parts[i].startswith("--"):
                    key = parts[i][2:]
                    i += 1
                    if key in [
                        "channels",
                        "lrs",
                        "strides",
                        "paddings",
                        "kernels",
                        "betas",
                    ]:
                        # Read the list values
                        values = []
                        while i < len(parts) and not parts[i].startswith("--"):
                            if parts[i].isdigit():
                                values.append(int(parts[i]))
                            elif parts[i].replace(".", "", 1).isdigit():
                                values.append(float(parts[i]))
                            else:
                                values.append(parts[i])
                            i += 1
                        load_params[key] = values
                    else:
                        if i < len(parts) and not parts[i].startswith("--"):
                            value = parts[i]
                            if value.isdigit():
                                load_params[key] = int(value)
                            elif value.replace(".", "", 1).isdigit():
                                load_params[key] = float(value)
                            elif value.lower() == "true":
                                load_params[key] = True
                            elif value.lower() == "false":
                                load_params[key] = False
                            elif value.startswith("[") and value.endswith("]"):
                                load_params[key] = eval(value)
                            else:
                                load_params[key] = value
                            i += 1
                        else:
                            load_params[key] = None
                else:
                    i += 1

    return load_params
