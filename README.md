<center><h1> RoI-Matching

THIS IS THE PREVIOUS VERSION, THE CURRENT VERSION AND DATASETS WILL BE RELEASED SOON! 

## Usage

- #### Installation

```bash
git clone https://github.com/pd162/RoI-Matching.git
cd RoI-Matching
pip install torch==<fit your cuda version>
pip install -r requirement.txt
```

- #### Preparation

https://rrc.cvc.uab.es/?ch=21&com=downloads

- #### Generate Labels

```
python vis_label.py
```

Remember edit the path!

- #### Training

```
CUDA_VISIBLE_DEVICES=0 python train_single_card.py
```

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2  --master_port 29501 --master_addr localhost train.py
```

- #### Eval

TBD

- #### Demo

TBD



