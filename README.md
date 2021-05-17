Vision Longformer for Object Detection 
=======
This project provides the source code for the object detection part of vision longformer paper. It is based on [detectron2](https://github.com/facebookresearch/detectron2).

[Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding](https://arxiv.org/abs/2103.15358)

The classification part of the code and checkpoints can be found [here](https://github.com/microsoft/vision-longformer). 

## Updates
- 03/29/2021: First version of [vision longformer paper](https://arxiv.org/abs/2103.15358) posted on Arxiv.  <br/>
- 05/17/2021: Performance improved by adding relative positional bias, inspired by [Swin Transformer](https://github.com/microsoft/Swin-Transformer)! First version of Object Detection code released. 


## Usage
Here is an example command for evaluating a pretrained vision-longformer small model on COCO
```
python -m pip install -e .

ln -s /mnt/data_storage datasets

DETECTRON2_DATASETS=datasets python train_net.py --num-gpus 1 --eval-only --config configs/msvit_maskrcnn_fpn_1x_small_sparse.yaml 
MODEL.TRANSFORMER.MSVIT.ARCH "l1,h3,d96,n1,s1,g1,p4,f7,a0_l2,h3,d192,n2,s1,g1,p2,f7,a0_l3,h6,d384,n8,s1,g1,p2,f7,a0_l4,h12,d768,n1,s1,g0,p2,f7,a0" 
SOLVER.AMP.ENABLED True 
MODEL.WEIGHTS /mnt/model_storage/msvit_det/visionlongformer/vilsmall/maskrcnn1x/model_final.pth
```


Here is an example training command for training the vision-longformer small model on COCO
```
python -m pip install -e .

ln -s /mnt/data_storage datasets

# convert the classification checkpoint into a detection checkpoint for initialization
python3 converter.py --source_model "/mnt/model_storage/msvit/visionlongformer/small1281_relative/model_best.pth"
--output_model msvit_pretrain.pth --config configs/msvit_maskrcnn_fpn_3xms_small_sparse.yaml
MODEL.TRANSFORMER.MSVIT.ARCH "l1,h3,d96,n1,s1,g1,p4,f7,a0_l2,h3,d192,n2,s1,g1,p2,f7,a0_l3,h6,d384,n8,s1,g1,p2,f7,a0_l4,h12,d768,n1,s1,g0,p2,f7,a0"

# train with the converted detection checkpoint as initialization
DETECTRON2_DATASETS=datasets python train_net.py --num-gpus 8 --config configs/msvit_maskrcnn_fpn_3xms_small_sparse.yaml
MODEL.WEIGHTS msvit_pretrain.pth MODEL.TRANSFORMER.DROP_PATH 0.2 MODEL.TRANSFORMER.MSVIT.ATTN_TYPE
longformerhand MODEL.TRANSFORMER.MSVIT.ARCH "l1,h3,d96,n1,s1,g1,p4,f7,a0_l2,h3,d192,n2,s1,g1,p2,f7,a0_l3,h6,d384,n8,s1,g1,p2,f7,a0_l4,h12,d768,n1,s1,g0,p2,f7,a0"
SOLVER.AMP.ENABLED True SOLVER.BASE_LR 1e-4 SOLVER.WEIGHT_DECAY 0.1 TEST.EVAL_PERIOD
7330 SOLVER.IMS_PER_BATCH 16
```

## Model Zoo on COCO

**Vision Longformer with relative positional bias**

| Backbone | Method | pretrain | drop_path | Lr Schd | box mAP | mask mAP | #params | FLOPs | checkpoints | log | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViL-Tiny | Mask R-CNN | ImageNet-1K | 0.05 | 1x | 41.4 | 38.1 | 26.9M | 145.6G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/viltiny/maskrcnn1x/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_tiny_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/viltiny/maskrcnn1x/stdout.txt) |
| ViL-Tiny | Mask R-CNN | ImageNet-1K | 0.1 | 3x | 44.2 | 40.6 | 26.9M | 145.6G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/viltiny/maskrcnn3x/model_final.pth) [config](configs/msvit_maskrcnn_fpn_3xms_tiny_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/viltiny/maskrcnn3x/stdout.txt) |
| ViL-Small | Mask R-CNN | ImageNet-1K | 0.2 | 1x | 44.9 | 41.1 | 45.0M | 218.3G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilsmall/maskrcnn1x/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_small_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilsmall/maskrcnn1x/stdout.txt) |
| ViL-Small | Mask R-CNN | ImageNet-1K | 0.2 | 3x | 47.1 | 42.7 | 45.0M | 218.3G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilsmall/maskrcnn3x/model_final.pth) [config](configs/msvit_maskrcnn_fpn_3xms_small_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilsmall/maskrcnn3x/stdout.txt) |
| ViL-Medium (D) | Mask R-CNN | ImageNet-21K | 0.2 | 1x | 47.6 | 43.0 | 60.1M | 293.8G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilmedium/maskrcnn1x/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_medium_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilmedium/maskrcnn1x/stdout.txt) |
| ViL-Medium (D) | Mask R-CNN | ImageNet-21K | 0.3 | 3x | 48.9 | 44.2 | 60.1M | 293.8G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilmedium/maskrcnn3x/model_final.pth) [config](configs/msvit_maskrcnn_fpn_3xms_medium_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilmedium/maskrcnn3x/stdout.txt) |
| ViL-Base (D) | Mask R-CNN | ImageNet-21K | 0.3 | 1x | 48.6 | 43.6 | 76.1M | 384.4G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilbase/maskrcnn1x/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_large_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilbase/maskrcnn1x/stdout.txt) |
| ViL-Base (D) | Mask R-CNN | ImageNet-21K | 0.3 | 3x | 49.6 | 44.5 | 76.1M | 384.4G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilbase/maskrcnn3x/model_final.pth) [config](configs/msvit_maskrcnn_fpn_3xms_large_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilbase/maskrcnn3x/stdout.txt) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViL-Tiny | RetinaNet | ImageNet-1K | 0.05 | 1x | 40.8 | -- | 16.64M | 182.7G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/viltiny/retinanet1x/model_final.pth) [config](configs/msvit_retina_fpn_1x.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/viltiny/retinanet1x/stdout.txt) |
| ViL-Tiny | RetinaNet | ImageNet-1K | 0.1 | 3x | 43.6 | -- | 16.64M | 182.7G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/viltiny/retinanet3x/model_final.pth) [config](configs/msvit_retina_fpn_3x_ms.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/viltiny/retinanet3x/stdout.txt) |
| ViL-Small | RetinaNet | ImageNet-1K | 0.1 | 1x | 44.2 | -- | 35.68M | 254.8G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilsmall/retinanet1x/model_final.pth) [config](configs/msvit_retina_fpn_1x.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilsmall/retinanet1x/stdout.txt) |
| ViL-Small | RetinaNet | ImageNet-1K | 0.2 | 3x | 45.9 | -- | 35.68M | 254.8G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilsmall/retinanet3x/model_final.pth) [config](configs/msvit_retina_fpn_3x_ms.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilsmall/retinanet3x/stdout.txt) |
| ViL-Medium (D) | RetinaNet | ImageNet-21K | 0.2 | 1x | 46.8 | -- | 50.77M | 330.4G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilmedium/retinanet1x/model_final.pth) [config](configs/msvit_retina_fpn_1x.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilmedium/retinanet1x/stdout.txt) |
| ViL-Medium (D) | RetinaNet | ImageNet-21K | 0.3 | 3x | 47.9 | -- | 50.77M | 330.4G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilmedium/retinanet3x/model_final.pth) [config](configs/msvit_retina_fpn_3x_ms.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilmedium/retinanet3x/stdout.txt) |
| ViL-Base (D) | RetinaNet | ImageNet-21K | 0.3 | 1x | 47.8 | -- | 66.74M | 420.9G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilbase/retinanet1x/model_final.pth) [config](configs/msvit_retina_fpn_1x.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilbase/retinanet1x/stdout.txt) |
| ViL-Base (D) | RetinaNet | ImageNet-21K | 0.3 | 3x | 48.6 | -- | 66.74M | 420.9G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilbase/retinanet3x/model_final.pth) [config](configs/msvit_retina_fpn_3x_ms.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/visionlongformer/vilbase/retinanet3x/stdout.txt) |

See more fine-grained results in Table 6 and Table 7 in the [Vision Longformer paper](https://arxiv.org/abs/2103.15358). We use weight decay 0.05 for all experiments, but search for best drop path in [0.05, 0.1, 0.2, 0.3]. 

**Comparison of various efficient attention mechanims with absolute positional embedding (Small size)**

| Backbone | Method | pretrain | drop_path | Lr Schd | box mAP | mask mAP | #params | FLOPs | Memory | checkpoints | log | 
| :---: | :---: | :---: | :---: | :---:  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| srformer/64 | Mask R-CNN | ImageNet-1K | 0.1 | 1x | 36.4 | 34.6 | 73.3M | 224.1G | 7.1G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/srformer64/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_srformer32_small_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/srformer64/stdout.txt) |
| srformer/32 | Mask R-CNN | ImageNet-1K | 0.1 | 1x | 39.9 | 37.3 | 51.5M | 268.3G | 13.6G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/srformer32/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_srformer64_small_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/srformer32/stdout.txt) |
| Partial srformer/32 | Mask R-CNN | ImageNet-1K | 0.1 | 1x | 42.4 | 39.0 | 46.8M | 352.1G | 22.6G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/srformerpartial/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_srformer32_small.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/srformerpartial/stdout.txt) |
| global | Mask R-CNN | ImageNet-1K | 0.1 | 1x | 34.8 | 33.4 | 45.2M | 226.4G | 7.6G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/global/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_gformer_small_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/global/stdout.txt) |
| Partial global | Mask R-CNN | ImageNet-1K | 0.1 | 1x | 42.5 | 39.2 | 45.1M | 326.5G | 20.1G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/globalpartial/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_gformer_small.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/globalpartial/stdout.txt) |
| performer | Mask R-CNN | ImageNet-1K | 0.1 | 1x | 36.1 | 34.3 | 45.0M | 251.5G | 8.4G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/performer/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_performer_small_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/performer/stdout.txt) |
| Partial performer | Mask R-CNN | ImageNet-1K | 0.05 | 1x | 42.3 | 39.1 | 45.0M | 343.7G | 20.0G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/performerpartial/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_performer_small.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/performerpartial/stdout.txt) |
| ViL | Mask R-CNN | ImageNet-1K | 0.1 | 1x | 42.9 | 39.6 | 45.0M | 218.3G | 7.4G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/longformer/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_small.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/longformer/stdout.txt) |
| Partial ViL | Mask R-CNN | ImageNet-1K | 0.1 | 1x | 43.3 | 39.8 | 45.0M | 326.8G | 19.5G | [ckpt](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/longformerpartial/model_final.pth) [config](configs/msvit_maskrcnn_fpn_1x_small_sparse.yaml) | [log](https://penzhanwu2.blob.core.windows.net/imagenet/msvit_det/attn_ablation_withape/longformerpartial/stdout.txt) |

We use weight decay 0.05 for all experiments, but search for best drop path in [0.05, 0.1, 0.2].
