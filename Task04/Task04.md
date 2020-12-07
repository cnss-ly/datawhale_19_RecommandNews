[toc]

## Task04：特征工程（3天）

- 打卡截止：12月03日23:59

- 开源内容：[Task04 天池新闻推荐入门赛之【特征工程】](http://datawhale.club/t/topic/201)

- 打卡链接：[https://shimo.im/forms/V52KtClLZN4CkAhj/fill 12](https://shimo.im/forms/V52KtClLZN4CkAhj/fill)

- 打卡结果：[https://shimo.im/sheets/16q8MDPYVGCXxrk7/MODOC/ 4](https://shimo.im/sheets/16q8MDPYVGCXxrk7/MODOC/

### 新闻推荐基础流程

#### 第一步：召回选出候选列表

新闻推荐的数据集是很庞大的，通过找回选出候选列表，将千、万甚至几十上百万的数据降低到几百、几十的数量级。
#### 第二步：数据集划分

数据集划分为训练集、交叉验证集、测试集。

#### 第三步：打标签

候选列表中命中的置为１，否则置为０。

#### 第四步：做特征工程

通过特征工程构建特征向量用作训练模型。
#### 第四步：训练模型

把做好的数据集喂给模型。



### 如何做特征工程

以目标为导向，如果选择比较热门的文章则可以通过一定时间文章点击量作为热度指标。

### word2vec

word2vec是谷歌提出一种word embedding 的工具或者算法集合，采用了两种模型(CBOW与skip-gram模型)与两种方法(负采样与层次softmax方法)的组合，比较常见的组合为 skip-gram+负采样方法。




