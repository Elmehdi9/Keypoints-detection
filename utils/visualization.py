from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import matplotlib.patheffects as path_effects
import cv2

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.pyplot import arrow
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib import colors

from scipy.signal import savgol_filter
from scipy.spatial import Voronoi

from shapely.geometry import Polygon



X_SIZE = 105
Y_SIZE = 68

BOX_HEIGHT = (16.5 * 2 + 7.32) / Y_SIZE * 100
BOX_WIDTH = 16.5 / X_SIZE * 100

GOAL = 7.32 / Y_SIZE * 100

GOAL_AREA_HEIGHT = 5.4864 * 2 / Y_SIZE * 100 + GOAL
GOAL_AREA_WIDTH = 5.4864 / X_SIZE * 100

SCALERS = np.array([X_SIZE / 100, Y_SIZE / 100])
pitch_polygon = Polygon(((0, 0), (0, 100), (100, 100), (100, 0)))


def visualize(**images):
    """PLot images in one row.

    Arguments:
        **images: images to plot
    Returns:
        
    Raises:
        
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()
