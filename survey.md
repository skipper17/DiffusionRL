# Survey

This survey focuses on the intersection of Diffusion Models and Reinforcement Learning (RL), with a particular emphasis on DDPO, Diffusion DPO, D3PO, and SPO. It covers detailed training methodologies and algorithms associated with these approaches.

## DDPO

There are two papers related, one is Training Diffusion Models with Reinforcement Learning and the other is Large-scale Reinforcement Learning for Diffusion Models.

### Method

Model the diffusion generation process, which can be considered a Markov Decision Process (MDP), as a reinforcement learning (RL) trajectory. Then, use some evaluation model on the image domain as return. Then, just apply the PPO algorithm.

这个方法像标准的rl一样，需要按照某个策略采样trajectory，此时有两种方案，一种是使用当前策略采样trajectory，之后梯度更新一次，使用新的策略采样trajectory。另外一种方法是使用一个指定的老模型采样trajectory，可以分小batch，多次更新模型，应用PPO算法。

在第一篇论文中，reward model通过LLaVA来对生成图像进行描述，通过BERTScore来进行跟原本生成图像的文本的相似度计算。该方法只是独立的使用唯一的reward model，比如前面的那个，或者直接使用LAION-predicted aesthetic quality那个模型。

在第二个工作中，探索了对advantage model的构建，也尝试了多个reward model一起使用的场景。reward轮询之后接上diffusion 默认loss。 有三个reward，一个是ImageReward和human feedback对齐，一个是衡量了生成的多样性，还有一个不太重要。

### Experiments

第一篇论文微调sd1.4， lr1e-5， bz 64， sample256， clip 1e-4，45类prompts。

第二篇论文 sd2，lr 2e-6，bz 16 per gpu，128 a100 1.5m DiffusionDB纯prompts。


## DiffusionDPO

主要是斯坦福的那篇 Diffusion DPO。注意这里diffusion dpo建模只有前缀是文本，后缀是图片A和图片B，之后再去沿图片生成的diffusion path得到相关的反馈，这个方法有一个问题是，最后的优化目标中并没有要求win和lose的eps相同，但是在他的实现中两个选择了相同的eps。

#### Experiments

reward model使用hpsv2，clip，aesthetic， pickscore
pickscore使用模型打伪标而非真实数据（beta5000， 2000steps）效果更好
beta=5000， 1000steps，lr=1e-8，16 a100 2048 bz

关于SFT： where an already task-tuned model does not benefit from preferred finetuning. 没用

测试下游任务

## SPO