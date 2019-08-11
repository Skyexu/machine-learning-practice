# 泰坦尼克生存预测

kaggle 链接： https://www.kaggle.com/c/titanic/overview/tutorials

目录说明：

- `./doc`: 一些参考的 kernel
- `./data`：比赛数据
- `./`

## 比赛描述

RMS泰坦尼克号沉没是历史上最臭名昭着的沉船之一。 1912年4月15日，在她的处女航中，泰坦尼克号在与冰山相撞后沉没，2224名乘客和机组人员中有1502人死亡。 这场耸人听闻的悲剧震惊了国际社会，并为船舶制定了更好的安全规定。

造成海难失事的原因之一是乘客和机组人员没有足够的救生艇。 尽管人们的幸存有一些运气因素，但有些人比其他人更容易生存，例如妇女，儿童和上流社会。

在这个挑战中，我们要求您完成对哪些人可能存活的分析。 特别是，我们要求您运用机器学习工具来预测哪些乘客幸免于悲剧。

## 评估

### 目标

预测乘客是否能幸存。对于每个测试样本，预测0或1的值。

### 评估指标

预测准确的百分比。[accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification)，精确率。

### 提交文件格式

418行加上标题，2列。

- PassengerId (sorted in any order)
- Survived (contains your binary predictions: 1 for survived, 0 for deceased)

```bash
PassengerId,Survived
 892,0
 893,1
 894,0
 Etc.
```



参考资料：

- https://www.kaggle.com/startupsci/titanic-data-science-solutions
- https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
  - [中文翻译](https://whyso.fun/2018/08/12/%E5%88%9D%E7%BA%A7%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E6%A1%86%E6%9E%B6%E2%80%94%E2%80%94%E4%BB%A5%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B%E5%8F%B7%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%BA%E4%BE%8B/)
- https://juejin.im/post/5c63bd7af265da2dd4274ef4#heading-13
- https://github.com/zmzhouXJTU/Titanic_Rescue_Prediction
- 数据分析：https://github.com/esskeetit0817/project-titanic/blob/master/project-titanic.ipynb

