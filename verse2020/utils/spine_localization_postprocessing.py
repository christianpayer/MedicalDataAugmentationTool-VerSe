
import numpy as np
import utils.np_image


def bb(image, transformation, image_spacing):
    """
    Calculate the bb of a heatmap in real world coordinates.
    :param image: The np array of the heatmap.
    :param transformation: The transformation.
    :return: (start, end) coordinate tuple.
    """
    image_thresholded = (np.squeeze(image / np.max(image)) > 0.5).astype(np.uint8)
    image_thresholded = utils.np_image.largest_connected_component(image_thresholded)
    start, end = utils.np_image.bounding_box(image_thresholded)
    start = np.flip(start.astype(np.float64) * np.array(image_spacing, np.float64))
    end = np.flip(end.astype(np.float64) * np.array(image_spacing, np.float64))
    start_transformed = transformation.TransformPoint(start)
    end_transformed = transformation.TransformPoint(end)
    return start_transformed, end_transformed


def bb_iou(bb0, bb1):
    """
    Calculate the bounding box intersection over union for two bounding boxes.
    :param bb0: Bounding box 0, (start, end) coordinate tuple.
    :param bb1: Bounding box 1, (start, end) coordinate tuple.
    :return: Intersection over union.
    """
    x_left = max(bb0[0][0], bb1[0][0])
    y_top = max(bb0[0][1], bb1[0][1])
    z_front = max(bb0[0][2], bb1[0][2])
    x_right = min(bb0[1][0], bb1[1][0])
    y_bottom = min(bb0[1][1], bb1[1][1])
    z_back = min(bb0[1][2], bb1[1][2])
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1) * (z_back - z_front + 1)

    # compute the area of both AABBs
    bb0_area = (bb0[1][0] - bb0[0][0] + 1) * (bb0[1][1] - bb0[0][1] + 1) * (bb0[1][2] - bb0[0][2] + 1)
    bb1_area = (bb1[1][0] - bb1[0][0] + 1) * (bb1[1][1] - bb1[0][1] + 1) * (bb1[1][2] - bb1[0][2] + 1)
    iou = intersection_area / float(bb0_area + bb1_area - intersection_area)

    return iou