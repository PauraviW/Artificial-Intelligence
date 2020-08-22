#!/usr/local/bin/python3

#

# Authors: [vmohan Vidit Mohaniya pwagh Pauravi Wagh vkvats Vibhas Vats]

#

# Mountain ridge finder

# Based on skeleton code by D. Crandall, Oct 2019

#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import numpy as np


# calculate "Edge strength map" of an image
# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels


def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale, 0, filtered_y)
    return sqrt(filtered_y ** 2)


def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):

        for t in range(int(max(y - int(thickness / 2), 0)), int(min(y + int(thickness / 2), image.size[1] - 1))):
            image.putpixel((x, t), color)

    return image


def viterbi(edge_strength, state, transition, emission):
    ridge = list()
    arg_max = []
    for key in range(edge_strength.shape[1]):

        l = list()
        temp = []
        for index in range(edge_strength.shape[0]):
            v01 = state[key] * transition[index + 1]
            temp.append(np.argmax(v01))
            l.append(emission[key][index] * max(v01))
        arg_max.append(temp)
        state[key + 1] = np.array(l)

    # Backtrack

    last_max_value = np.argmax(state[edge_strength.shape[1]])
    for i in range(len(arg_max)-1, -1, -1):
        ridge.append(arg_max[i][last_max_value])
        last_max_value = arg_max[i][last_max_value]

    ridge.reverse()
    return ridge

# main program

(input_filename, gt_row, gt_col) = sys.argv[1:]
# load in image
input_image = Image.open(input_filename)
# compute edge strength mask
edge_strength = edge_strength(input_image)
imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))
# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
rez = map(list, zip(*edge_strength))

################## Emission values calculation #################


edge_sum = np.sum(edge_strength, axis=0)
emission = (edge_strength / edge_sum).transpose()
state = dict()
state[0] = emission[0]
emission = emission * edge_strength.shape[1]

#########transistion values calculation #############


transition = {}
for j in range(1, edge_strength.shape[0] + 1):
    distance = [(5 + (col_index - j) ** 2) for col_index in range(1, edge_strength.shape[0] + 1)]

    tot_dist = sum(distance)

    factor = [(tot_dist / ec) for ec in distance]

    transition[j] = np.array([(f / sum(factor)) for f in factor])

############ part wise solution ############


# simple solution part 2.1
prob = edge_strength / edge_sum
# prob = 255 * edge_strength / (amax(edge_strength))
simple_ridge = np.argmax(prob, axis=0)
imageio.imwrite("output_simple.jpg", draw_edge(input_image, simple_ridge, (0, 0, 255), 5))

# viterbi part 2.2
input_image = Image.open(input_filename)
viterbi_ridge = viterbi(edge_strength, state, transition, emission)
imageio.imwrite("output_map.jpg", draw_edge(input_image, viterbi_ridge, (255, 0, 0), 5))

# # viterbi part 2.3 - human intervention

input_image = Image.open(input_filename)
emission[:, int(gt_col)] = 0
emission[int(gt_row), int(gt_col)] = 0
human_ridge = viterbi(edge_strength, state, transition, emission)
imageio.imwrite("output_human.jpg", draw_edge(input_image, human_ridge, (0, 255, 0), 5))
