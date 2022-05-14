from __future__ import division
import random

import numpy as np
import torch


def mean_shift(points, scores, embedding, alpha=0.6):
    points_embedding = list(zip(points, embedding, scores))
    point_group = []
    while len(points_embedding) > 0:
        center = random.choice(range(len(points_embedding)))
        center = points_embedding[center][1]
        for _ in range(5):
            selected = [
                x[1]
                for x in points_embedding
                if np.linalg.norm(x[1] - center) <= alpha
            ]
            center = np.mean(selected, axis=0)
        selected = [(x[0], x[2])
                    for x in points_embedding
                    if np.linalg.norm(x[1] - center) <= alpha]
        point_group.append(selected)
        points_embedding = [
            x for x in points_embedding
            if np.linalg.norm(x[1] - center) > alpha
        ]
    return point_group


def nms(points_score_group):
    outputs = []
    for lps in points_score_group:
        ys = set(x[0][1] for x in lps)
        output = []
        for y in ys:
            points = sorted([x for x in lps if x[0][1] == y],
                            key=lambda x: x[1],
                            reverse=True)
            output.append(points[0])
        outputs.append(output)
    return outputs
