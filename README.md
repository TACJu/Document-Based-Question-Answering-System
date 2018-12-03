# Document-Based-Question-Answering-System

### Task

Based on given Chinese documents, develop a simplified automatic question answering system (DBQA) and  for each sentence in the document output a decimal number ranging in `[0, 1]` representing how likely the sentence is containing the correct answer to the given question.

This task is very similar to [MSRA's Beauty of Programming Contest](https://studentclub.msra.cn/bop2017/rules/qualification).

### Training/Validation data

Training set: 264416 samples

Validataion set: 39997 samples

Training/Validation data format:

```
俄罗斯贝加尔湖的面积有多大?\t 贝加尔湖，中国古代称为北海，位于俄罗斯西伯利亚的南部。\t 0 
俄罗斯贝加尔湖的面积有多大?\t 贝加尔湖是世界上最深，容量最大的淡水湖。\t 0 
俄罗斯贝加尔湖的面积有多大?\t 贝加尔湖贝加尔湖是世界上最深和蓄水量最大的淡水湖。\t 0 
俄罗斯贝加尔湖的面积有多大?\t 它位于布里亚特共和国(Buryatiya) 和伊尔库茨克州(Irkutsk) 境内。\t 0 
俄罗斯贝加尔湖的面积有多大?\t 湖型狭长弯曲，宛如一弯新月，所以又有“月亮湖”之称。\t 0 
俄罗斯贝加尔湖的面积有多大?\t 湖长636公里，平均宽48公里，最宽79.4公里，面积3.15万平方公里。\t 1 
俄罗斯贝加尔湖的面积有多大?\t 贝加尔湖湖水澄澈清冽，且稳定透明(透明度达40.8米)，为世界第二。\t 0
```

### Output of test data

Each line should contain only one number representing the score of relevance between a question and a sentence from the given document.

These numbers are used for ranking all sentences in given document by toolkit.

### Evaluation

1. Mean Average Precision
2. Mean Reciprocal Rank

### Roadmap

- [ ] Word embedding
- [ ] Simple LSTM benchmark
- [ ] ...

### Team members

Ju He, Dongwei Xiang, Yuzhang Hu, Xu Song