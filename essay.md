## MDP模型构建
> 参考文献：Q-learning算法优化的SVDPP推荐算法 **[[1]](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=JSJC202102007&uniplatform=NZKPT&v=g9BjGJf5ZLWNr6XbnULgXnHYCkdwLO%25mmd2FmBomvFc%25mmd2Fxda8Fhbmlg8D2YBEaRjrPq4GO)**。 

### 马尔可夫决策过程的描述  

马尔可夫决策过程可以用五元组进行描述：
$$
<\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma>
$$

论文的核心观点在于：用户过去的评分数据对当前的评分有显著影响,将用户对电影的喜好隐式地反映在时间戳中,有助于得到更精确的结果。  
论文采用 MovieLens 1M 数据集作为研究对象，因此需要将用户在不同时间戳下对电影的评分转换成五元组以构造马尔科夫决策过程：
1. 状态空间$\mathcal{S}$。论文将用户u在时间t下对电影的评分记为状态$s^{(u)}_t$ ，因为数据集中用户对电影的评分是［1，5］区间内的5个整数，所以$s^{(u)}_t$的范围为［1，5］，所有时间戳下的状态$s^{(u)}_t$构成了状态空间$\mathcal{S}$。
2. 动作空间$\mathcal{A}$。考虑到用户u在时间t下看了电影并给出了评分$s^{(u)}_t$ ，该评分会影响其（t+1）时间对电影的评分$s^{(u)}_{t + 1}$，所以将$a^{(u)}_t$记为从$s^{(u)}_t$到$s^{(u)}_{t+1}$的动作；所有时刻的动作$a^{(u)}_t$构成了动作空间$\mathcal{A}$。
3. 状态转移概率$\mathcal{P}$。用户u在状态$s^{(u)}_t$下采取的动作$a^{(u)}_t$是由时间戳决定的，认为状态之间的转移概率也是确定的，即$a^{(u)}_{t} =s^{(u)}_{t + 1}$，$P(s^{(u)}_{t + 1}|s^{(u)}_{t},a^{(u)}_{t + 1})=1$。
4. 奖励函数$\mathcal{R}$。其作为一个状态中完成某个动作所获得的奖励，是强化学习所需的关键要素。论文定义了：
$$
R(s^{(u)}_{t},a^{(u)}_{t} ) = s^{(u)}_{t+2}-\hat{r}_{ui}
$$

    其中：$\hat{r}_{ui}$表示用SVD 或 SVDPP 模型计算出的用户 u 对电影 i 的预测评分。
5. 折扣因子$\gamma$。每次动作会产生对应的奖励，但是同一用户观看电影时，越是后期的奖励折扣越大。设$0\leq\gamma<1$。  

### 状态表与马尔可夫链生成  
由上述马尔科夫决策过程可知，一个状态转移到下一个状态的动作对应下一个时间电影的评分，通过这个过程可将 MovieLens 1M 数据集处理为下表所示的形式。
<div align=center><img src ="https://cdn.jsdelivr.net/gh/Jeff-974/my-git1/images/table1.png"/></div>

将表 1 的数据按照时间戳排序，生成的马尔可夫链如下：
$$\begin{aligned}
&5\stackrel{3}{\rightarrow}3\stackrel{4}{\rightarrow}4\stackrel{3}{\rightarrow}3\\ 
&4\stackrel{5}{\rightarrow}5\\ 
&1\stackrel{3}{\rightarrow}3\stackrel{4}{\rightarrow}4\stackrel{4}{\rightarrow}4\\ 
&3\stackrel{5}{\rightarrow}5\stackrel{2}{\rightarrow}2\\ 
&4\stackrel{1}{\rightarrow}1\stackrel{4}{\rightarrow}4\stackrel{2}{\rightarrow}2
\end{aligned}
$$  
<img width="350" alt="屏幕截图 2023-10-22 113853" src="https://github.com/Jeff-974/my-git1/assets/81134373/b0ffa9f1-a226-4dcc-bf71-cbb8a42f670a">
