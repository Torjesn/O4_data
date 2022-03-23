from re import A, I
import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Compute intersection
    intersection_x = min(prediction_box[2], gt_box[2]) - max(prediction_box[0], gt_box[0])
    if intersection_x < 0:
        intersection_x = 0  

    intersection_y = min(prediction_box[3], gt_box[3]) - max(prediction_box[1], gt_box[1])
    if intersection_y < 0:
        intersection_y = 0      
    intersection = intersection_x * intersection_y

    # Compute union
    area_predection = (prediction_box[2] - prediction_box[0])* (prediction_box[3] - prediction_box[1])
    area_gt = (gt_box[2] - gt_box[0])* (gt_box[3] - gt_box[1])
    union = area_predection + area_gt - intersection 

    iou = intersection/union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    precision = 1
    if (num_tp+num_fp > 0):
        precision = num_tp/(num_tp+num_fp)
    return precision

def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    recall = 0
    if (num_tp+num_fn > 0):
        recall = num_tp/(num_tp+num_fn)
    return recall

def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    num_gt = gt_boxes.shape[0]
    num_pred = prediction_boxes.shape[0]

    matched  = []

    for pr_n in range(num_pred):
        for gt_n in range(num_gt):
            iou = calculate_iou(prediction_boxes[pr_n], gt_boxes[gt_n])
            if iou >= iou_threshold:
                matched.append([iou, gt_n, pr_n])

    matched = np.array(matched)
    if matched.size > 0:
        matched = matched[matched[:, 0].argsort()[::-1]]


    matched_gt = []
    matched_pred = []
    taken_gt = []
    taken_pred = []

    for match in matched:
        if (not match[1] in taken_gt) and (not match[2] in taken_pred):
            taken_gt.append(match[1])
            matched_gt.append(gt_boxes[int(match[1])])

            taken_pred.append(match[2])
            matched_pred.append(prediction_boxes[int(match[2])])

            if(len(taken_gt) == num_gt):
               break
                
    return np.array(matched_pred), np.array(matched_gt)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    matched_gt, _ = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    pos = prediction_boxes.shape[0]
    true = gt_boxes.shape[0]
    true_pos = matched_gt.shape[0]
    false_pos = pos - true_pos
    false_neg = true - true_pos

    dict = {"true_pos": true_pos, "false_pos": false_pos, "false_neg": false_neg}
    return dict


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.

    """
    true_pos_tot = 0
    false_pos_tot = 0
    false_neg_tot = 0
    for im_pred_boxes, im_gt_boxes in zip(all_prediction_boxes, all_gt_boxes):
        dict = calculate_individual_image_result(im_pred_boxes, im_gt_boxes, iou_threshold)
        
        true_pos_tot += dict["true_pos"]
        false_pos_tot += dict["false_pos"]
        false_neg_tot += dict["false_neg"]


    precision = calculate_precision(true_pos_tot, false_pos_tot, false_neg_tot)
    recall = calculate_recall(true_pos_tot, false_pos_tot, false_neg_tot)

    return precision, recall

def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)

    precisions = []
    recalls = []
    for threshold in confidence_thresholds:
        confident_preds = []
        for boxes, scores in zip(all_prediction_boxes, confidence_scores):
            confident_scores =  scores > threshold
            im_pred = []
            for box, b in zip(boxes, confident_scores):
                if b:
                   im_pred.append(box)
            np_im_pred = np.array(im_pred) 
            confident_preds.append(np_im_pred)
        pr, rc = calculate_precision_recall_all_images(confident_preds, all_gt_boxes, iou_threshold)
        precisions.append(pr)
        recalls.append(rc)
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """

    print("Precision sum is: " + str(np.sum(precisions)))
    print("Recall sum is: " + str(np.sum(recalls)))
    #sort the precisions
    pres_rec = np.vstack((precisions, recalls))
    s = pres_rec[1, :].argsort()
    pres_rec = pres_rec[:,s]

    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    max_pres_tot = 0

    for level in recall_levels:
        index_over_level = np.where(pres_rec[1] >= level)
        pres = pres_rec[0,index_over_level]
        max_pres = 0
        if pres.size > 0:
            max_pres = np.amax(pres)
        max_pres_tot += max_pres

    # YOUR CODE HERE
    average_precision = max_pres_tot / 11.0
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
