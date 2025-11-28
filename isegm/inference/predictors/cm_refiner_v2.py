import torch
from .base import BasePredictor


class CMRefinerV2Predictor(BasePredictor):

    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)
        if not self.net.with_prev_mask:
            raise ValueError("CMRefinerV2Predictor 要求其模型 (CMRefinerModel_V2) 的 with_prev_mask=True")



    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)

        num_clicks = len(clicks_lists[0])
        # gate = 0 if num_clicks == 1 else 1
        gate = 0 if num_clicks <= 1 else 1

        model_output = self.net(image_nd, points_nd, gate=gate)

        return model_output