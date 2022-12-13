# Paddle Implementation of DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local and Global Features (ICCV 2021)


## Training

- Pytorch Version: There are increasing inquiries for Pytorch codes and models,  we have had our pytorch models and codes ready [at this url](https://github.com/feymanpriv/DOLG).


- We are happy to see from a Chinese technical media post (https://mp.weixin.qq.com/s/7B3hZUpLtTt8NcGt0c-77w) that our DOLG has been adopted as one of building blocks to the Kaggle21 landmark competition winner solution . In this post, third-party Pytorch code snippets of DOLG are also presented.


## Evaluation

```
cd revisitop && python example_evaluate.py
``` 

**modified results from torch weights**


|  			 					 | Roxf-M | +1M | Rpar-M | +1M   | Roxf-H | +1M  | Rpar-H | +1M  |
|:------------------------------:|:------:|:---:|:------:|:-----:|:------:|:----:|:------:|:----:|
|  DOLG-R50(with query cropping) |  81.20 |71.36| 90.07  | 78.99 |  62.55 |47.34 | 79.20  | 59.75|
|  DOLG-R101(with query cropping)|  82.37 |73.63| 90.97  | 80.44 |  64.93 |51.57 | 81.71  | 62.95|
|                                                                                                |
|  DOLG-R50(w/o query cropping)  |  82.38 |77.78| 90.94  | 82.16 |  62.92 | 55.48| 80.48  | 65.77| 
|  DOLG-R101(w/o query cropping) |  83.22 |78.96| 91.64  | 82.89 |  64.83 | 57.86| 82.56  | 67.34|


## Weights

- [R101-DOLG](https://pan.baidu.com/s/1gLqpq4nqK4-tLpuf-5tcEQ) (a25u)   [R50-DOLG](https://pan.baidu.com/s/1wA0bR5YC-LLge0ZkU5lR2w) (1on8)

- [Infer-model-weights](https://pan.baidu.com/s/1qz5qMjtIgXxFMsnG07Kwhg) (f3gf)


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

