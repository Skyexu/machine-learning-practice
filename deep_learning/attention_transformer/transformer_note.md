文章整理参考资料

1. https://luweikxy.github.io/machine-learning-notes/#/natural-language-processing/self-attention-and-transformer/attention-is-all-you-need/attention-is-all-you-need 优先参考，原理顺序。训练与decoder部分不是太详细
2. https://blog.csdn.net/wuzhongqiang/article/details/104414239?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161512240816780357259240%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=161512240816780357259240&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_v1~rank_blog_v1-1-104414239.pc_v1_rank_blog_v1&utm_term=Attention+is+all 原理参考合并优秀描述，训练过程描述

需要进一步理解的内容

- 数学理解：[为什么 dot-product attention 需要被 scaled？](https://blog.csdn.net/qq_37430422/article/details/105042303)
- self_attention 中 q、k、v的作用
- 位置编码的作用，为什么论文这样计算，有什么好处

mask 作用

- 

**Attention和self-attention的区别**

以Encoder-Decoder框架为例，输入Source和输出Target内容是不一样的，比如对于英-中机器翻译来说，Source是英文句子，Target是对应的翻译出的中文句子，Attention发生在Target的元素Query和Source中的所有元素之间。

Self Attention，指的不是Target和Source之间的Attention机制，而是Source内部元素之间或者Target内部元素之间发生的Attention机制，也可以理解为Target=Source这种特殊情况下的Attention。

两者具体计算过程是一样的，只是计算对象发生了变化而已。



