## Get Similar Chinese Words and Sentences

Requirements:

-  Python >= 3.5
- NumPy

To do List:

- [x] Chinese Words
- [ ] Reimplementation GloVe Model
- [ ] Chinese Sentence 

Usage:

```bash
git clone https://github.com/HoratioJSY/cn-words.git
cd cn-words
python inference.py --word 机器学习
```

### For examples

- Get the similar words

Set Flag 「word」to chose target and search similar words:

```bash
>>> python inference.py --word 机器学习
机器学习 is close to: 模式识别、数据挖掘、深度学习、图形学、人工智能、神经网络、信号处理、运筹学、信息论、地理信息系统、数字图像处理、微分方程、面向对象、并行计算、概率论、故障诊断、生物信息学、数理统计
Inference time: 0.7688698768615723
```

```bash
>>> python inference.py --word 谢霆锋                                                                                
谢霆锋 is close to: 张学友、国语专辑、周杰伦、刘德华、王力宏、张惠妹、郭富城、陈奕迅、林俊杰、孙燕姿、梁咏琪、林忆莲、梅艳芳、任贤齐、容祖儿、谭咏麟、张韶涵、陈慧琳
Inference time: 0.7493560314178467
```

Set Flag 「top_k」to chose the numbers of nearest words:

```bash
>>> python inference.py --word 聚精会神 --top_k 12
聚精会神 is close to: 全神贯注、专心致志、凝神、扎扎实实、一心一意、认认真真、伟大旗帜、认真、真抓实干、专心、解放思想、集中精力
Inference time: 0.7596039772033691
```

- Get the similarity of two words

Base on word Vector to get Cosine similarity:

```bash
>>> python inference.py --word 微积分/概率论
Similarity is:  0.8017578125
Inference time: 0.0002090930938720703

>>> python inference.py --word 微积分/物理
Similarity is:  0.404052734375
Inference time: 0.00020599365234375

>>> python inference.py --word 微积分/文科生
Similarity is:  0.2337646484375
Inference time: 0.0002460479736328125
```

- Words analogies


```bash
>>> python inference.py --word 卷积+深度学习
卷积 + 深度学习 is close to: 卷积神经网络、深度学习、循环神经网络、神经网络、微分方程、模式识别、自适应、傅里、数据挖掘、时域、差分、信号处理、滤波、频域、多项式、运算符、非线性、随机变量
Inference time: 0.7463030815124512

>>> python inference.py --word 摩托车-单车
摩托车 - 单车 is close to: 摩托车、汽车、轿车、客车、拖拉机、摩托、机动车、零部件、农用、三轮、卡车、变速器、电视机、跑车、小轿车、柴油、奥迪、汽油
Inference time: 0.7612090110778809
```

- Adding New Words


There is a simple  and intuitive way to add new word to vocabulary. If flag 「add_vocabulary」is missing,  output the testing results:

```bash
>>> python inference.py --add_word '残差网络=0.3*卷积神经网络+0.3*残差+0.3*图像识别+0.1*人工智能'
残差网络 is close to: 深度学习、循环神经网络、模式识别、神经网络、数据挖掘、自适应、信号处理、数字信号、时域、差分、微分方程、频域、图形学、人工智能、随机变量、线性规划、滤波、数字图像处理

```

If flag 「add_vocabulary」 is True, a new word vector is generated, and it will be added to vocabulary:

```bash
>>> python3 inference.py --add_word '残差网络=0.3*卷积神经网络+0.3*残差+0.4*图像识别' --add_vocabulary True
残差网络 is close to: 循环神经网络、深度学习、模式识别、神经网络、数字信号、时域、自适应、频域、数据挖掘、差分、信号处理
Successfully update vocabulary: 残差网络
```
