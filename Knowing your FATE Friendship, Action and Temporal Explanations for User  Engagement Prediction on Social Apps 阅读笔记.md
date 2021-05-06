# Knowing your FATE: Friendship, Action and Temporal Explanations for User  Engagement Prediction on Social Apps 阅读笔记

本文发表于KDD2020，单位为宾夕法尼亚州立大学，一作为Xianfeng Tang

## 1.研究问题

本文研究了**具有可解释性的社交网络用户参与度（粘性）预测**问题

## 2. 使用方法及框架

* 根据业务的不同要求，给出了不同场景下的用户参与度的预测。
* 设计了一个端到端的神经网络模型框架FATE，其主要考虑了友谊（**F**riendships），用户行为（user **A**ction），时间动态（**T**emporal dynamics）并具有可解释性（**E**xplainable engagements）。
* 基础方法：GNN，TLSTM以及mixture attention mechanism

## 3. 背景、意义以及以往研究

* 大部分社交网络应用开发的首要任务时吸引并维持庞大的用户基础，提高用户粘性。
* 以往研究的缺陷在于：
  * 建模用户参与度时没有模拟友谊依赖性以及用户之间的交互
  * 参与目标不同，例如广告团队以点击率的预测为目标，用户增加团队关注不同功能的使用趋势。
  * 现有方法侧重于预测用户参与度，并未给出解释。来说明用户为什么参与或者不参与。

* 主要贡献在于：
  * 可解释的用户参与预测
  * 为不同业务场景的用户参与度设计了灵活的定义
  * FATE提出
  * 使用Snapchat的两个真实数据集评估了FATE，误差减少10%，运行时间提升20%

* 相关工作的缺陷：
  * 具有优越性能的神经网络模型是黑盒的，其结果不具备可解释性，即使是输入有意义的用户活动特征。
  * 可解释的GNN或者RNN模型分类：（1）经过训练的模型的事后解释；（2）学习预测的近似值。缺陷在于通常是为特定的深度学习架构设计的专用解释模型。本文试图探索**分层深度学习的细致的**解释。

## 4. 模型框架细节

### 4.1 基本定义

* 符号定义

| 符号                  | 解释                                    |
| --------------------- | --------------------------------------- |
| $u$                   | 一个注册的社交网络用户                  |
| $T$                   | 过去的一段时间点                        |
| $t$                   | 时间变量                                |
| $v$                   | 用户*u*的一位朋友                       |
| $\mathbf{G}_{t}^{u}$  | 用户图                                  |
| $\mathcal{V}_{t}^{u}$ | 用户节点的集合                          |
| $N_t(u)$              | 与用户$u$有关的用户群体                 |
| $\mathcal{E}_{t}^{u}$ | 友谊关系                                |
| $\mathbf{X}_{t}^{u}$  | 节点特征矩阵表示用户行为特征            |
| $\mathbf{E}_{t}^{u}$  | 边特征表示用户的交互特征                |
| $K$                   | 将用户特征分为K类用于表示特定的用户特征 |

* 用户图定义：

  用户图由四部分组成，分别是节点集合、边集合、节点特征矩阵、边特征矩阵，其中节点表示用户，边表示用户之间的交互行为。本文将用户的节点特征分为了K类，节点特征随$t$变化，每时刻由K个特征连接而成。由图像可以看出节点特征建模用户的操作包括阅读、打电话、拍照、听音乐等；而用户的交互行为包括发图片、聊天、点赞、互粉、取消互粉等。

![image-20210505115636302](C:\Users\47899\AppData\Roaming\Typora\typora-user-images\image-20210505115636302.png)

* 用户参与度定义：

  用户参与度的应该因业务场景而异，例如Facebook使用登录频率衡量用户参与度而SnapChat使用发信息的频率来衡量用户参与度。因此，本文应给出灵活的定义。其使用对未来一段时间的用户利益指标的期望来定义用户参与度。$\left.e_{t}^{u}=\mathbb{E}(\mathcal{M}(u, \tau) \mid \tau \in[t, t+\Delta t])\right)$ 其中$M$是指感兴趣的指标。

* 用户参与度的解释：

  本文中的解释是给出用户行为、时间动态和友谊的**重要性**。从局部（针对个别用户）和全局（整个人群和用户程序）给出解释。局部重要性定义$A^{u}$（1）对于用户行为重要性，给出该用户对于K种交互行为的重要性分数。（2）时序重要性$T^{u}$，对于指定的时间段表示了用户对于$k$操作的重要性分数。（3）友谊重要性$F^{u}$，对于用户$u$的朋友$v$们，其对于用户$u$的用户参与度的重要性。因此针对全局来说其定义为：（1）用户行为$A^{*}$;（2）全局时序特征$T^{*}$

* 问题形式化
  * 对于任意用户，预测其用户参与度得分$e_{T}^{u}$以及解释$A^{u},T^{u},F^{u}$.
  * 生成全局解释$A^{*},T^{*}$.

### 4.2 FATE框架

* 总体框架图

![image-20210505123835889](C:\Users\47899\AppData\Roaming\Typora\typora-user-images\image-20210505123835889.png)

（1）使用GCN模块生成友谊特征和互动特征。

（2）使用TLSTM捕捉时序信息。

（3）使用混合注意力机制给出解释。

* 友谊模块（Friendship Module）

  ​	本文实验后发现社交网络用户仅仅跟所有朋友中的一部分（20%）联系较为紧密如图4：![image-20210505125001834](C:\Users\47899\AppData\Roaming\Typora\typora-user-images\image-20210505125001834.png)

  ​	传统GCN将根据邻居节点的特征更新自身特征表示：$\tilde{\mathbf{x}}^{u}=\sigma\left(\sum_{v \in \mathcal{N}(v)} \mathbf{x}^{v} \mathbf{W}\right)$

  但是传统CGN将每个特征混合到了一起无法解释每个子特征的重要性，因此，本文提出了*tensor-based GCN*，其公式为：$\tilde{\mathbf{x}}^{u}=\sigma\left(\sum_{v \in \mathcal{N}(v)} \mathbf{x}^{v} \otimes \mathcal{W}\right)$
  
  ![image-20210505190842681](C:\Users\47899\AppData\Roaming\Typora\typora-user-images\image-20210505190842681.png)
  
  ​	针对每种特征都有对应的节点特征矩阵，针对每种特征的特征矩阵使用对用的权重参数对其进行转化，使其长度变为1，之后将所有的特征矩阵连接起来。每种特征的意思是对应每种用户之间的交互动作。本文使用了两层TGCN，也就是最远获取了朋友的朋友的特征：
  $$
  \tilde{\mathbf{X}}=\sigma\left(\hat{\mathrm{A}} \sigma\left(\hat{\mathrm{A}} \mathrm{X} \otimes \mathcal{W}_{0}\right) \otimes \mathcal{W}_{1}\right)
  $$
  ​	之后本文使用了友谊注意力机制（friendship attention mechanism）来衡量用户$u$不同朋友对于其的重要性，其使用了用户$v$的节点特征以及用户$u$和用户$v$的关系特征注意力分数计算方程为：
  $$
  \alpha_{v}=\frac{\exp \left(\phi\left(\tilde{\mathbf{x}}^{v} \oplus \mathbf{e}^{v}\right)\right)}{\sum_{v \in \mathcal{N}(u)} \exp \left(\phi\left(\tilde{\mathbf{x}}^{v} \oplus \mathbf{e}^{v}\right)\right)}
  $$
  ​	之后获得使用朋友重要性调整过后的特征矩阵：
  $$
  \hat{\mathbf{x}}=\sum_{v \in \mathcal{N}(u)} \alpha_{v} \tilde{\mathbf{x}}^{v}
  $$
  ​	将未调整的矩阵和调整后的矩阵连接：$\mathrm{g}^{u}=\tilde{\mathbf{x}}^{u} \oplus \hat{\mathbf{x}}=\left[\tilde{\mathbf{x}}_{1}^{u} \oplus \hat{\mathbf{x}}_{1}, \cdots, \tilde{\mathbf{x}}_{K}^{u} \oplus \hat{\mathbf{x}}_{K}\right]$
  
  之后获得每一个时刻的图$\mathbf{g}_{1}^{u}, \cdots \mathbf{g}_{T}^{u}$
  

* 时序模块

  类似于LSTM，作者使用了改进的TLSTM，其特征更新公式为：
  $$
  \mathbf{f}_{t}=\sigma\left(\mathbf{g}_{t}^{u} \otimes \mathcal{U}_{f}+\mathbf{h}_{t-1} \otimes \mathcal{U}_{f}^{\mathrm{h}}+\mathbf{b}_{f}\right) \\
  \mathbf{i}_{t}=\sigma\left(\mathbf{g}_{t}^{u} \otimes \mathcal{U}_{i}+\mathbf{h}_{t-1} \otimes \mathcal{U}_{i}^{\mathrm{h}}+\mathbf{b}_{i}\right) \\
  \mathbf{o}_{t}=\sigma\left(\mathbf{g}_{t}^{u} \otimes \mathcal{U}_{o}+\mathbf{h}_{t-1} \otimes \mathcal{U}_{o}^{\mathrm{h}}+\mathbf{b}_{o}\right) \\
  \mathbf{c}_{t}=\mathbf{f}_{t} \odot \mathbf{c}_{t-1}+\mathbf{i}_{t} \odot \tanh \left(\mathbf{g}_{t}^{u} \otimes \mathcal{U}_{c}+\mathbf{h}_{t-1} \otimes \mathcal{U}_{c}^{\mathrm{h}}+\mathbf{b}_{c}\right) \\
  \mathbf{h}_{t}=\mathbf{o}_{t} \odot \tanh \left(\mathbf{c}_{t}\right)
  $$

* 用户参与分的生成

  ​	本文的问题中由三种潜在变量$Z_A,Z_J,Z_I$分别是用户操作、时序信息以及友谊,本文使用了联合概率来计算用户参与概率矩阵：

  ![image-20210505201216817](C:\Users\47899\AppData\Roaming\Typora\typora-user-images\image-20210505201216817.png)

  T时刻的用户参与概率主要分为四个部分：节点嵌入特征，友谊注意力，时序注意力以及用户行为注意力。此处的概率值均可以解释为注意力机制，其中时序注意力概率计算公式为：
  $$
  \beta_{t, k}=p\left(z_{J}=t \mid z_{A}=k,\left\{\mathbf{h}_{*, k}\right\}\right)=\frac{\exp \left(\varphi_{k}\left(\mathbf{h}_{t, k}\right)\right)}{\sum_{\tau=1}^{T} \exp \left(\varphi_{k}\left(\mathbf{h}_{\tau, k}\right)\right)}
  $$
  用户概率为：
  $$
  p\left(z_{A}=k \mid\left\{\mathbf{h}_{*}\right\}\right)=\frac{\exp \left(\phi\left(\mathbf{a}_{k} \oplus \mathbf{h}_{T, k}\right)\right)}{\sum_{K=1}^{K} \exp \left(\phi\left(\mathbf{a}_{\kappa} \oplus \mathbf{h}_{T, \kappa}\right)\right)}​
  $$
  此外本文还使用了用户高斯分布来估计每个用户行为对于用户参与度的贡献。
  $$
  p\left(e_{T}\right)=\sum_{k=1}^{K} N\left(\mu_{k}, s d_{k}\right) \cdot p\left(z_{A}=k \mid\left\{\mathbf{h}_{*}\right\}\right) .
  $$

## 5 实验评估

### 5.1 基本设置 

* 数据集：使用了两个大型不同城市的SnapChat的数据集，两个数据集分别处于不同的大陆。共使用了6周的数据，前三周用来训练，后三周用来测试。输入两周的数据并对下一周进行预测。
* 评估指标：RMSE, MAE, MAPE
* 对比方法：LR, XGBoost, MLP, LSTM, GCN, Temporal GCN-LSTM
* 消融实验：
  * $FATE_{ts}$:使用原始的GCN+LSTM。
  * $FATE_{fnd}$: 去除友谊模块。
  * $FATE_{tmp}$: 去除时序模块。
  * $FATE_{int}$:去除交互特征。

### 5.2 可解释性

主要以重要性的表以及图来展示可解释性。