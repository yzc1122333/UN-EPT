## Dependencies
* PyTorch 1.5+ and torchvision 0.6+
* Python 3.7
* Install [MMCV](https://mmcv.readthedocs.io/en/latest/) 

```
pip install mmcv
```
* Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/install.md) :
```
pip install mmsegmentation # install the latest release
```

* Compiling CUDA operators of [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)


## Data 
### ADE20k
Prepare the data like this.

```
path/to/ADEChallengeData2016/
  images/
    training/
    validation/

  annotations/ 
    training/
    validation/

  dt_offset/
    training/
    validation/
```

## Usage 
### Train on ADE20k
**1. The default is for multi-gpu, DistributedDataParallel training due to SyncBN.**

```
python -m torch.distributed.launch --nproc_per_node=8 \ # specify gpu number
--master_port=29500  \
train.py  --launcher pytorch \
--config_file /path/to/config_file 
```

<!-- **A runnable command (multi-gpu train):**
- specify the ```data_root``` in the config ```configs/resV1c50_ss_deform_tr2+2_480x480_adamW_step_640k_ade20k.py```;
- log dir will be created in ```./work_dirs```.

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 train.py  --launcher pytorch \
--config_file configs/resV1c50_ss_deform_tr2+2_480x480_adamW_step_640k_ade20k.py 
``` -->

<!-- **2. For single-gpu, change the type of 'norm_cfg' to 'BN' in the config.**

```
python train.py --config_file /path/to/config_file \
--work_dir /path/to/log_dir \
``` -->



<!-- **3. I try to take 'H/8*W/8 x 256' as the vector size for decoder input and remain stride 32 for the backbone.**

In the config, ```num_queries=4096``` and ```dec_stride=8``` are specified.

It can run with single gpu (one sample on it). (```norm_cfg = dict(type='BN', requires_grad=True)```)
```
python train.py --config_file ./configs/resV1c50_32x_dec8x_tr2+2_512x512_adamW_step_640k_ade20k.py
```

But, it got the ```CUDA error: an illegal memory access was encountered``` when trained with multi-gpu (```norm_cfg = dict(type='SyncBN', requires_grad=True)```):

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 train.py  --launcher pytorch \
--config_file ./configs/resV1c50_32x_dec8x_tr2+2_512x512_adamW_step_640k_ade20k.py
``` -->


### Evaluate on ADE20k
<!-- 1. directly resize image and ground truth to 512x512 by bilinear interpolation and nearest interpolation, respectively. ('test_mode': 'direct_resize')

The default is for single-gpu evaluation with batch size being 1.
```
python evaluate.py --eval \
--config_file configs/resV1c50_segtr2_512x512_ade20k.yaml \
--checkpoint log/checkpoint_0131.pth.tar \
--data_root /path/to/ADEChallengeData2016
``` -->

1. Use slide mode, the crop size is the same as the model input size (now 480 x 480). 

```
# single-gpu testing
python test.py --checkpoint /path/to/checkpoint \
--eval mIoU \
[--out ${RESULT_FILE}] [--show] \
--aug-test \ # for multi-scale flip aug

# multi-gpu testing (4 gpus, 1 sample per gpu)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
test.py  --launcher pytorch --eval mIoU \
--aug-test \ # for multi-scale flip aug
```

<!-- ### Using Longformer

1. install longformer (when using tvm mode, the compiled file only supports CUDA 10.0 with pyTorch 1.2)
```
pip install git+https://github.com/allenai/longformer.git
```

2. reinstall mmcv and mmsegmentation
```
pip install mmcv==1.1.0
pip install mmsegmentation==0.5.0
``` -->

### Evaluate on Cityscapes
1. on validation set.

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
test.py  --launcher pytorch  \
--aug-test \ # for multi-scale flip aug
--out results.pkl --eval mIoU cityscapes
```

2. on test set. save images for submitting to the server.

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
test.py  --launcher pytorch  \
--aug-test \ # for multi-scale flip aug
--format-only --eval-options "imgfile_prefix=./test_results"
```






 