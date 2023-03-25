# ManiCLIP

![Teaser image](teaser.png)

## Dataset

During the training phase, we do not need any data, except the 40-category face [attributes](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). During the testing phase, the text data can be obtained from [Multi-Modal-CelebA-HQ](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset).

## Pretrained models

The pretrained models can be downloaded from this [link](https://entuedu-my.sharepoint.com/:u:/g/personal/hao005_e_ntu_edu_sg/EVJ4RLcEpgNLoHSZ6zAuCxMB1MWFrLz0CSGwKU-T9Bu3tA?e=uzHiZw).

## Training

You can train new face editing networks using `train.py`.

```.bash
python train.py --epochs 30 --loss_id_weight 0.05 --loss_w_norm_weight 0.1 --loss_clip_weight 1.0 --loss_face_norm_weight 0.05 --loss_minmaxentropy_weight 0.2 --loss_face_bg_weight 1 --task_name name --decouple --part_sample_num 3
```

## Generation

To generate edited images based on language:

```.bash
python generate.py --model_path pretrained/pretrained_edit_model.pth.tar --text "this person has grey hair. he has mustache." --gen_num 5
```
