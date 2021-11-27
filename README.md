# Unofficial Paddlepaddle implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

This repository is built upon [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch) and [PaddleViT] (https://github.com/BR-IDL/PaddleViT), thanks very much!


Now, we implement the pretrain and finetune process according to the paper, but still **can't guarantee** the performance reported in the paper can be reproduced! 

## Difference

### `shuffle` and `unshuffle`

`shuffle` and `unshuffle` operations don't seem to be directly accessible in pytorch, so we use another method to realize this process:
+ For `shuffle`, we use the method of randomly generating mask-map (14x14) in BEiT, where `mask=0` illustrates keeping the token, `mask=1` denotes dropping the token (not participating caculation in encoder). Then all visible tokens (`mask=0`) are fed into encoder network.
+ For `unshuffle`, we get the postion embeddings (with adding the shared mask token) of all masked tokens according to the mask-map and then concate them with the visible tokens (from encoder), and feed them into the decoder network to recontrust.

### sine-cosine positional embeddings

The positional embeddings mentioned in the paper are `sine-cosine` version. And we adopt the implemention of [here](https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31), but it seems like a 1-D embeddings not 2-D's. So we don't know what effect it will bring.


## TODO
- [ ] perfomance report
- [ ] add the `cls` token in the encoder
- [ ] knn and linear prob
- [ ] ...

## Setup

```
pip install -r requirements.txt
```

## Run
  One-click Pretrain,Finetune,Visualization of reconstruction,visit to Baidu AI Studio Project 
  [Masked AutoEncoders（MAE) ](https://aistudio.baidu.com/aistudio/projectdetail/2798001?contributionType=1) 


## Result
   comming soon ...
 
So if one can fininsh it, please feel free to report it in the issue or push a PR, thank you!

And your star is my motivation, thank u~
