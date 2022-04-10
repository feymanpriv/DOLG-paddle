# Paddle Implementation of DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local and Global Features (ICCV 2021)


- According to some policy of our organization, we have to had our PaddlePaddle code well prepared before we release it. Currently, we are working on it and our code will come soon after we get through the approval process. 

- We are happy to see from a Chinese technical media post (https://mp.weixin.qq.com/s/7B3hZUpLtTt8NcGt0c-77w) that our DOLG has been adopted as one of building blocks to the Kaggle21 landmark competition winner solution . In this post, third-party Pytorch code snippets of DOLG are presented, but we did not officially check their correctness. If you are interested in reproducing DOLG using Pytorch, please contact yangminbupt@outlook.com in case you have any question.


## Training

- Pytorch Version: There are increasing inquiries for Pytorch codes and models,  we have had our pytorch models and codes ready.  If you really need them for RESEARCH purpose,  please contact the authors in private.


## Evaluation

```
cd revisitop && python example_evaluate.py
``` 

**modified results from torch weights**

|  			 					 | Roxf-M | +1M | Rpar-M | +1M   | Roxf-H | +1M  | Rpar-H | +1M  |
|:------------------------------:|:------:|:---:|:------:|:-----:|:------:|:----:|:------:|:----:|
|  DOLG-R50(with query cropping) |  81.20 |  -- | 90.07  |       |  62.55 |  --  | 79.20  |      |
|  DOLG-R101(with query cropping)|  82.37 |  -- | 90.97  |       |  64.93 |  --  | 81.71  |      |
|                                                                                                |
|  DOLG-R50(w/o query cropping)  |  82.38 |     | 90.94  |       |  62.92 |      | 80.48  |      | 
|  DOLG-R101(w/o query cropping) |  83.22 |     | 91.64  |       |  64.83 |      | 82.56  |      |



## Weights

- [R101-DOLG](https://pan.baidu.com/s/1gLqpq4nqK4-tLpuf-5tcEQ) (a25u)   [R50-DOLG](https://pan.baidu.com/s/1wA0bR5YC-LLge0ZkU5lR2w) (1on8)


## Citation

If the project helps your research, please consider citing our paper as follows.

```BibTeX
@InProceedings{Yang_2021_ICCV,
    author={Yang, Min and He, Dongliang and Fan, Miao and Shi, Baorong and Xue, Xuetong and Li, Fu and Ding, Errui and Huang, Jizhou},
    title={DOLG: Single-Stage Image Retrieval With Deep Orthogonal Fusion of Local and Global Features},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month={October},
    year={2021},
    pages={11772-11781}
}

