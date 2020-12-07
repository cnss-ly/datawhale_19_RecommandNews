[toc]

## Task05：排序模型+模型融合（3天）

- 打卡截止：12月06日23:59
- 开源内容：[Task05 天池新闻推荐入门赛之【排序模型+模型融合】](http://datawhale.club/t/topic/202)
- 打卡链接：[https://shimo.im/forms/TRxG2raTij0iVHCI/fill 19](https://shimo.im/forms/TRxG2raTij0iVHCI/fill)
- 打卡结果：[https://shimo.im/sheets/1d3aVvb0N5I92Pqg/MODOC/ 9](https://shimo.im/sheets/1d3aVvb0N5I92Pqg/MODOC/)

### LightGBM简介

GBDT (Gradient Boosting Decision Tree) 主要思想是利用弱分类器（决策树）迭代训练以得到最优模型，通常被用于多分类、点击率预测、搜索排序等任务；在各种数据挖掘竞赛中也是大杀器。而LightGBM（Light Gradient Boosting Machine）是一个实现GBDT算法的框架，支持高效率的并行训练，并且具有更快的训练速度、更低的内存消耗、更好的准确率、支持分布式可以快速处理海量数据等优点。

#### LightGBM提出的动机

常用的机器学习算法，例如神经网络等算法，都可以以mini-batch的方式训练，训练数据的大小不会受到内存限制。而GBDT在每一次迭代的时候，都需要遍历整个训练数据多次。如果把整个训练数据装进内存则会限制训练数据的大小；如果不装进内存，反复地读写训练数据又会消耗非常大的时间。尤其面对工业级海量的数据，普通的GBDT算法是不能满足其需求的。LightGBM提出的主要原因就是为了解决GBDT在海量数据遇到的问题，让GBDT可以更好更快地用于工业实践。

#### XGBoost的缺点

首先，空间消耗大。这样的算法需要保存数据的特征值，还保存了特征排序的结果（例如，为了后续快速的计算分割点，保存了排序后的索引），这就需要消耗训练数据两倍的内存。

其次，时间上也有较大的开销，在遍历每一个分割点的时候，都需要进行分裂增益的计算，消耗的代价大。

最后，对cache优化不友好。在预排序后，特征对梯度的访问是一种随机访问，并且不同的特征访问的顺序不一样，无法对cache进行优化。同时，在每一层长树的时候，需要随机访问一个行索引到叶子索引的数组，并且不同特征访问的顺序也不一样，也会造成较大的cache miss。

#### LightGBM的优化

为了避免上述XGBoost的缺陷，并且能够在不损害准确率的条件下加快GBDT模型的训练速度，lightGBM在传统的GBDT算法上进行了如下优化：

基于Histogram的决策树算法。

单边梯度采样 Gradient-based One-Side Sampling(GOSS)：使用GOSS可以减少大量只具有小梯度的数据实例，这样在计算信息增益的时候只利用剩下的具有高梯度的数据就可以了，相比XGBoost遍历所有特征值节省了不少时间和空间上的开销。

互斥特征捆绑 Exclusive Feature Bundling(EFB)：使用EFB可以将许多互斥的特征绑定为一个特征，这样达到了降维的目的。

带深度限制的Leaf-wise的叶子生长策略：大多数GBDT工具使用低效的按层生长 (level-wise)的决策树生长策略，因为它不加区分的对待同一层的叶子，带来了很多没必要的开销。实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。

LightGBM使用了带有深度限制的按叶子生长 (leaf-wise) 算法。

直接支持类别特征(Categorical Feature)

支持高效并行

Cache命中率优化

#### lightGBM具体实现

待续