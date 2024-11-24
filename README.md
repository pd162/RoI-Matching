<center><h1> RoI-Matching


## Usage

- #### Installation

```bash
git clone https://github.com/pd162/RoI-Matching.git
cd RoI-Matching
pip install torch==<fit your cuda version>
pip install -r requirement.txt
```

- #### Preparation

[SVRD]
https://rrc.cvc.uab.es/?ch=21&com=downloads

[CDLA]
https://github.com/buptlihang/CDLA

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

```
CUDA_VISIBLE_DEVICES=0 python test.py
```

- #### Demo

TBD


## Acknowledge


