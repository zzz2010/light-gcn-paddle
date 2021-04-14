







### 团队成员： 小木桌，chf544062970， 江江123
### 复现结果要点
1、使用的数据集、模型文件及完整的复现代码

- `数据集` ：[79027](https://aistudio.baidu.com/aistudio/datasetdetail/79027)  数据文件是在LightGCN 官方github上面下载的 
- `完整的复现代码` 在这个项目的folder：light-gcn-paddle (论文实现代码)，  paddorch (提供pytorch接口的paddle实现)

关于我写的torch接口代码请参考 [pytorch 转 paddle 心得](https://blog.csdn.net/weixin_48733317/article/details/108176827)
有兴趣了解的朋友可以看我在这个[视频](https://aistudio.baidu.com/aistudio/education/lessonvideo/698277)的Paddorch介绍（10分钟位置开始），
之前我用paddorch库复现了3个GAN类别的项目。


值得注意的是虽然说这个是LightGCN的paddle版本，但你基本上看不到paddle api接口，因为都被我们在Paddorch库中重新封装了， 所以代码看起来就跟torch一样 




2、提供具体详细的说明文档(或notebook)，内容包括:

(1) 数据准备与预处理步骤

- 数据直接挂载和解压，没有其他预处理步骤
 
 
 
 
(2) 训练脚本/代码，最好包含训练一个epoch的运行日志

- 在下面的cells 包含10个epoch的训练示例，和所有训练的命令行（完整训练记录参考下面） 
- `main.py` 是入口文件， 跟官方代码一样接口，参考[pytorch repo](https://github.com/gusye1234/LightGCN-PyTorch)
- 我们用了下面3个项目来训练3个不同的数据集，请点击每个项目来查看完整的训练记录
1. [Amazon-book](https://aistudio.baidu.com/aistudio/clusterprojectdetail/1781368) 脚本训练项目
2. [Gowalla](https://aistudio.baidu.com/aistudio/projectdetail/1792581) notebook 训练项目
3. [Yelp2018](https://aistudio.baidu.com/aistudio/projectdetail/1796108) notebook 训练项目




(3) 测试脚本/代码，必须包含评估得到最终精度的运行日志

- 原来的官方代码没有独立的测试脚本，测试是包含在training script里面，每10 epoch evaluation 一次，所以在训练的log中
看到 recall， precision  ndcg 都是测试集的metric , k=20 
- 我单独写一个测试脚本 `python eval_model.py --dataset [dataset] --path [model_file]` 输出recall@20, NDCG@20, Precision@20




(4) 最终精度，如精度相比源码有提升，需要说明精度提升用到的方法与技巧(不可更换网络主体结构，不可将测试集用于训练)

#### Amazon book Recall@20是0.0393， 验收要求 0.0384
#### Amazon book NDCG@20是0.0304， 验收要求 0.0298
#### Yelp2018 Recall@20是x， 验收要求 0.0631
#### Yelp2018 NDCG@20是X， 验收要求 0.0515
#### Gowalla Recall@20是0.1790， 验收要求 0.177
#### Gowalla NDCG@20是0.1506， 验收要求 0.1492

我们根据论文尝试了不同的超参数，和论文Fig.4中的LightGCN-Single模式，发现2 layers的LightGCN-Single模式在两个dataset上表现更佳。

另外，我们通过两个阶段不同学习率的训练，第一个阶段用大的learning rate 1e-3和默认的batch size (1024). 第二阶段（finetune）用很小的learning rate 1e-4和较大的batch size (10240)。

其他我们尝试过的不同超参数产生的模型可以在[数据集80647](https://aistudio.baidu.com/aistudio/datasetdetail/80647)中下载



(5) 其它学员觉得需要说明的地方
- 记得安装paddorch库， `cd paddorch;pip install .`
- 关键点可以看我实现的paddorch.sparse.mm 函数,用了nn.functional.embedding 

3、上传最终训练好的模型文件
- 在`light-gcn-paddle/paddle_pretrained_models`  

4、如评估结果保存在json文件中，可上传最终评估得到的json文件
- 没有生成json文件， 但在测试脚本的最后输出需要的评估指标






===========================================================================
#### Update

2020-09:
* Change the print format of each epoch
* Add Cpp Extension in  `code/sources/`  for negative sampling. To use the extension, please install `pybind11` and `cppimport` under your environment

---

## LightGCN-pytorch

This is the Pytorch implementation for our SIGIR 2020 paper:

>SIGIR 2020. Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).

Author: Prof. Xiangnan He (staff.ustc.edu.cn/~hexn/)

(Also see Tensorflow [implementation](https://github.com/kuandeng/LightGCN))

## Introduction

In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation. We propose a new model named LightGCN,including only the most essential component in GCN—neighborhood aggregation—for collaborative filtering



## Enviroment Requirement

`pip install -r requirements.txt`



## Dataset

We provide three processed datasets: Gowalla, Yelp2018 and Amazon-book and one small dataset LastFM.

see more in `dataloader.py`

## An example to run a 3-layer LightGCN

run LightGCN on **Gowalla** dataset:

* change base directory

Change `ROOT_PATH` in `code/world.py`

* command

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64`

* log output

```shell
...
======================
EPOCH[5/1000]
BPR[sample time][16.2=15.84+0.42]
[saved][[BPR[aver loss1.128e-01]]
[0;30;43m[TEST][0m
{'precision': array([0.03315359]), 'recall': array([0.10711388]), 'ndcg': array([0.08940792])}
[TOTAL TIME] 35.9975962638855
...
======================
EPOCH[116/1000]
BPR[sample time][16.9=16.60+0.45]
[saved][[BPR[aver loss2.056e-02]]
[TOTAL TIME] 30.99874997138977
...
```

*NOTE*:

1. Even though we offer the code to split user-item matrix for matrix multiplication, we strongly suggest you don't enable it since it will extremely slow down the training speed.
2. If you feel the test process is slow, try to increase the ` testbatch` and enable `multicore`(Windows system may encounter problems with `multicore` option enabled)
3. Use `tensorboard` option, it's good.
4. Since we fix the seed(`--seed=2020` ) of `numpy` and `torch` in the beginning, if you run the command as we do above, you should have the exact output log despite the running time (check your output of *epoch 5* and *epoch 116*).


## Extend:
* If you want to run lightGCN on your own dataset, you should go to `dataloader.py`, and implement a dataloader inherited from `BasicDataset`.  Then register it in `register.py`.
* If you want to run your own models on the datasets we offer, you should go to `model.py`, and implement a model inherited from `BasicModel`.  Then register it in `register.py`.
* If you want to run your own sampling methods on the datasets and models we offer, you should go to `Procedure.py`, and implement a function. Then modify the corresponding code in `main.py`


## Results
*all metrics is under top-20*

***pytorch* version results** (stop at 1000 epochs):

(*for seed=2020*)

* gowalla:

|             | Recall | ndcg | precision |
| ----------- | ---------------------------- | ----------------- | ---- |
| **layer=1** | 0.1687               | 0.1417    | 0.05106 |
| **layer=2** | 0.1786                     | 0.1524    | 0.05456 |
| **layer=3** | 0.1824                | 0.1547 | 0.05589 |
| **layer=4** | 0.1825                 | 0.1537       | 0.05576 |

* yelp2018

|             | Recall | ndcg | precision |
| ----------- | ---------------------------- | ----------------- | ---- |
| **layer=1** | 0.05604     | 0.04557 | 0.02519 |
| **layer=2** | 0.05988               | 0.04956 | 0.0271 |
| **layer=3** | 0.06347          | 0.05238 | 0.0285 |
| **layer=4** | 0.06515                | 0.05325 | 0.02917 |

