# SRAFS

This is the official implementation of the paper "Interactive Image Segmentation with Cross-Modality Vision Transformers".

### <p align="center"> Semantic Reweighting and Attention Field Supervision for Interactive Image Segmentation
<br>

<div align="center">
  Kun&nbsp;Li</a> <b>&middot;</b>
  George&nbsp;Vosselman</a> <b>&middot;</b>
  Michael Ying&nbsp;Yang</a>
  <br> <br>
  <a href="https://openaccess.thecvf.com/content/ICCV2023W/NIVT/html/Li_Interactive_Image_Segmentation_with_Cross-Modality_Vision_Transformers_ICCVW_2023_paper.html" target="_blank">Paper</a>
</div>
</br>

<div align=center><img src="assets/iCMFormer.png" /></div>

### Abstract

In recent years, interactive image segmentation methods have commonly adopted an iterative refinement paradigm, where the predicted mask from the previous iteration is used as guidance to progressively improve subsequent segmentation results. Although some approaches enhance the representation of user interactions by modulating the previous mask, such informative cues are typically fused only at the input layer. As the information propagates deeper into the network, its influence gradually diminishes, making it difficult to continuously guide the learning of high-level semantic features. In addition, existing models often rely on self-attention mechanisms to implicitly infer user intent without explicit supervisory signals, which may result in ambiguous attention maps and lead to increased corrective interactions from the user. To address the limitations of shallow fusion, we design an Adaptive Guided Fusion module that injects modulated corrective signals not only at the input layer but also into intermediate layers of the Transformer backbone, ensuring that user intent actively guides feature learning across multiple semantic depths. Furthermore, to obtain more discriminative feature representations, we introduce an Attention Field Supervision loss that directly regularizes self-attention maps by penalizing spatial overlap between foreground and background attention fields, encouraging the network to learn highly discriminative attention patterns. Extensive experiments on multiple benchmark datasets demonstrate that the proposed Semantic Reweighting and Attention Field Supervision (SRAFS) interactive segmentation model achieves superior performance compared with other state-of-the-art methods.

### Preparations

PyTorch 0.9.0, torchvision==0.9.0, torch 1.8.0.

```
pip3 install -r requirements.txt
```

### Download

The datasets for training and validation can be downloaded by following: [RITM Github](https://github.com/saic-vul/ritm_interactive_segmentation)

The pre-trained models are coming soon.

### Evaluation

Before evaluation, please download the datasets and models and configure the path in configs.yml.

The following script will start validation with the default hyperparameters:

```
python scripts/evaluate_model.py CMRefiner-V2 \
--gpu=0 \
--checkpoint=./weights/SRAFS_cocolvis.pth \
--eval-mode=cvpr \
--datasets=GrabCut,Berkeley,DAVIS,SBD
```

### Training

Before training, please download the pre-trained weights (click to download: [ViT](https://github.com/facebookresearch/mae) and [Swin](https://github.com/microsoft/Swin-Transformer)).

Use the following code to train a base model on coco+lvis dataset:

```
python train.py ./models/segformerB3_S2_cclvs.py \
--batch-size=6 \
--ngpus=1
```

## Acknowledgement
Here, we thank so much for these great works:  [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation)
