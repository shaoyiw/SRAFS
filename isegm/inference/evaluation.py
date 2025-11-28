from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, attn_callback=None, **kwargs):
    all_ious ,all_bious, all_assds= [], [], []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        for object_id in sample.objects_ids:
            _, sample_ious, sample_bious, sample_assds, _ = evaluate_sample(sample.image, sample.image_name, sample.gt_mask(object_id), predictor,
                                                sample_id=index, attn_callback=attn_callback, **kwargs)
            all_ious.append(sample_ious)
            all_bious.append(sample_bious)
            all_assds.append(sample_assds)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, all_bious, all_assds, elapsed_time


def evaluate_sample(image, image_name, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None, dilation_ratio=0.02, attn_callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list, biou_list, assd_list = [], [], []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if attn_callback is not None:
                attn_callback(predictor, sample_id, click_indx)

            iou = utils.get_iou(gt_mask, pred_mask)
            biou = utils.get_biou(gt_mask, pred_mask, dilation_ratio=dilation_ratio)
            assd = utils.get_assd(gt_mask, pred_mask)

            if callback is not None:
                # callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)
                callback(image, gt_mask, pred_probs, iou, image_name, click_indx, clicker.clicks_list)

            ious_list.append(iou)
            biou_list.append(biou)
            assd_list.append(assd)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), np.array(biou_list, dtype=np.float32), np.array(
            assd_list, dtype=np.float32), pred_probs
