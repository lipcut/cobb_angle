from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def landmarks_rearrange(landmarks: np.ndarray) -> np.ndarray:
    r"""rearrange the points so that the output array will be a the consective array of
    top left (tl), top right (tr), bottom left (bl), bottom right (br).
    i.e.
    output = [
        tl,
        tr,
        bl,
        br,
        ...
     ]

    where all points are 2d arrays.

    we do this by separating the points by y indexes to left and right points (2 points for each),
    then separating each pairs to top and bottom ones.
    """

    # make the array more structural so that it's easier to work with (e.g. sorting)
    # i.e.
    # [[x y], ...] -> [(x, y), ...]
    new_points = np.array(
        [(x, y) for x, y in landmarks], dtype=[("x", np.float32), ("y", np.float32)]
    )
    # sort to the lefts and the rights
    boxes = np.sort(new_points.reshape(17, 4), order="x", axis=1)
    # sort to the tops and the buttoms
    boxes = np.sort(boxes.reshape(17, 2, 2), order="y", axis=2)

    # now we have an array looks like this:
    # [[[tl bl],
    #   [tr br]],...]
    #
    # but we want this:
    # [[[tl tr],
    #   [bl br]],...]
    #
    # if there's no adjustment, then when it's flatten, the order won't be correct
    boxes = np.transpose(boxes, (0, 2, 1)).reshape(17, 4)
    # sort the arrage of points by the heights of their centers
    center_y = np.mean(boxes["y"], axis=1)
    boxes = boxes[np.argsort(center_y, axis=0)].reshape(-1)
    return np.array([[x, y] for x, y in boxes])


def landmarks_resize(
    landmarks: torch.Tensor,
    orig_dim: Tuple[int, ...] | torch.Tensor,
    new_dim: Tuple[int, ...] | torch.Tensor,
    padding: Optional[Tuple[int, ...] | torch.Tensor] = None,
) -> torch.Tensor:
    orig_dim = torch.Tensor(orig_dim)
    new_dim = torch.Tensor(new_dim)
    if padding is None:
        padding = torch.zeros(len(orig_dim.size()))
    else:
        padding = torch.Tensor(padding)
    t_landmarks = landmarks + padding
    t_landmarks = t_landmarks * (new_dim / (orig_dim + 2 * padding))
    return t_landmarks


def draw_spinal(pts, out_image):
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    for i in range(4):
        cv2.circle(out_image, (int(pts[i, 0]), int(pts[i, 1])), 3, colors[i], 1, 1)
        cv2.putText(
            out_image,
            "{}".format(i + 1),
            (int(pts[i, 0]), int(pts[i, 1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 0),
            1,
            1,
        )
    for i, j in zip([0, 1, 2, 3], [1, 2, 3, 0]):
        cv2.line(
            out_image,
            (int(pts[i, 0]), int(pts[i, 1])),
            (int(pts[j, 0]), int(pts[j, 1])),
            color=colors[i],
            thickness=1,
            lineType=1,
        )
    return out_image
