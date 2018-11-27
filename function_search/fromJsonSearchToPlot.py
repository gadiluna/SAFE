# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#
import matplotlib.pyplot as plt
import json
import math
import numpy as np
from multiprocessing import Pool


def find_dcg(element_list):
    dcg_score = 0.0
    for j, sim in enumerate(element_list):
        dcg_score += float(sim) / math.log(j + 2)
    return dcg_score


def count_ones(element_list):
    return len([x for x in element_list if x == 1])


def extract_info(file_1):
    with open(file_1, 'r') as f:
        data1 = json.load(f)

    performance1 = []

    average_recall_k1 = []
    precision_at_k1 = []

    for f_index in range(0, len(data1)):

        f1 = data1[f_index][0]
        pf1 = data1[f_index][1]

        tp1 = []

        recall_p1 = []
        precision_p1 = []
        # we start from 1 to remove ourselves
        for k in range(1, 200):
            cut1 = f1[0:k]
            dcg1 = find_dcg(cut1)
            ideal1 = find_dcg(([1] * (pf1) + [0] * (k - pf1))[0:k])

            p1k = float(count_ones(cut1))

            tp1.append(dcg1 / ideal1)
            recall_p1.append(p1k / pf1)
            precision_p1.append(p1k / k)

        performance1.append(tp1)
        average_recall_k1.append(recall_p1)
        precision_at_k1.append(precision_p1)

    avg_p1 = np.average(performance1, axis=0)
    avg_p10 = np.average(average_recall_k1, axis=0)
    average_precision = np.average(precision_at_k1, axis=0)
    return avg_p1, avg_p10, average_precision


def print_graph(info1, file_name, label_y, title_1, p):
    fig, ax = plt.subplots()
    ax.plot(range(0, len(info1)), info1, color='b', label=title_1)
    ax.legend(loc=p, shadow=True, fontsize='x-large')
    plt.xlabel("Number of Nearest Results")
    plt.ylabel(label_y)
    fname = file_name
    plt.savefig(fname)
    plt.close(fname)


def compare_and_print(file):
    filename = file.split('_')[0] + '_' + file.split('_')[1]
    t_short = filename
    label_1 = t_short + '_' + file.split('_')[3]

    avg_p1, recall_p1, precision1 = extract_info(file)

    fname = filename + '_nDCG.pdf'
    print_graph(avg_p1, fname, 'nDCG', label_1, 'upper right')

    fname = filename + '_recall.pdf'
    print_graph(recall_p1, fname, 'Recall', label_1, 'lower right')

    fname = filename + '_precision.pdf'
    print_graph(precision1, fname, 'Precision', label_1, 'upper right')

    return avg_p1, recall_p1, precision1


e1 = 'embeddings_safe'

opt = ['O0', 'O1', 'O2', 'O3']
compilers = ['gcc-7', 'gcc-4.8', 'clang-6.0', 'clang-4.0']
values = []
for o in opt:
    for c in compilers:
        f0 = '' + c + '_' + o + '_' + e1 + '_top200.json'
        values.append(f0)

p = Pool(4)
result = p.map(compare_and_print, values)

avg_p1 = []
recal_p1 = []
pre_p1 = []

avg_p2 = []
recal_p2 = []
pre_p2 = []

for t in result:
    avg_p1.append(t[0])
    recal_p1.append(t[1])
    pre_p1.append(t[2])

avg_p1 = np.average(avg_p1, axis=0)
recal_p1 = np.average(recal_p1, axis=0)
pre_p1 = np.average(pre_p1, axis=0)

print_graph(avg_p1[0:20], 'nDCG.pdf', 'normalized DCG', 'SAFE', 'upper right')
print_graph(recal_p1, 'recall.pdf', 'recall', 'SAFE', 'lower right')
print_graph(pre_p1[0:20], 'precision.pdf', 'precision', 'SAFE', 'upper right')
