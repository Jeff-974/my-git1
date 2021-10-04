## CHAMPTER 5: Monte Carlo Methods  
### 5.1 Monte Carlo Prediction  

> 蒙特卡罗方法的思想：利用在第二章的状态值方法中的思想，采用大数定律，某个状态的值是后继状态<b>累积的折扣回报</b>的期望。估计该值的方法是求得<b>观测</b>该状态对应的回报的均值，随着观测的次数的增多，均值应当收敛至状态值。这就是蒙特卡罗方法(MC method)。
> 本章聚焦于first-visit MC method，即计算首次访问状态s的回报的均值来作为s的状态值。


- first-visit MC算法:  
1. 初始化:  
$\pi \leftarrow$ 待评估的策略;  
$V \leftarrow$ 任意值;  
$Returns(s) \leftarrow$ 空表, $\forall s\in \mathcal{S}$;  
2. 循环`while True`:  
- 利用$\pi$产生一次实验;  
- `for s in `本次实验出现的所有状态:  
    - $G \leftarrow$ 首次出现s之后得到的回报
    - $Returns(s)$`.append`($G$)  
    - $V(s)\leftarrow$`np.mean`($Returns(s)$)  

当首次访问s的次数趋近于无穷时，first-visit MC收敛于$v_{\pi}(s)$。  
蒙特卡罗算法重要的性质是：对每个状态的估计都是**独立**的，不依赖于对其他状态的估计。这也是与DP方法的不同点。  
### 5.2 Monte Carlo Estimation of Action Values  
> 用蒙特卡罗算法应该来估计状态值函数$v_\pi(s)$还是行为值函数$q_\pi(s,a)$呢？
> 答案是：在环境模型未知的情况下，估计行为值函数$q_\pi(s,a)$。因为该情况下状态值函数是**不充分**的。

因此，对策略的评估，就需要考虑所有状态和每个状态下的所有动作，即是所有的$(s,a)\ pairs$，在起始的时候概率的赋值都应当大于0（**探索性初始化**）。这样才能确保进行无数次实验的同时，每个状态的每个动作也会接受无数次实验来得以估计所有的行为值函数。  
### 5.3 Monte Carlo Control  
- first-visit MC ES:  
1. 初始化:  
$\forall s\in\mathcal{S},a\in\mathcal{A}$:  
    - $\pi(s) \leftarrow$ 任意策略;  
    - $Q(s,a) \leftarrow$ 任意值;  
    - $Returns(s,a) \leftarrow$ 空表,  
2. 循环`while True`:  
   - 利用$\pi$产生一次实验，该实验从$S_0\in\mathcal{S},A_0\in\mathcal{A(\mathcal{S_0})}$开始，每个s-a对都应该符合探索性条件;  
   - `for (s,a) in `本次实验出现的所有状态-动作对:  
      - $G \leftarrow$ 首次出现(s,a)对 之后得到的回报
      - $Returns(s,a)$`.append`($G$)  
      - $Q(s,a)\leftarrow$`np.mean`($Returns(s,a)$)  
   - `for s in` 本实验出现的所有状态:  
      - $\pi(s)\leftarrow \underset{a}{\argmax} Q(s,a)$ (greedy)

### 5.4 On-Policy Predicts
> 探索性初始化理论上可行，但实际上却是不太可能实现的。因此，只能保证智能体继续可以选择所有的动作。
> On-Policy Predicts: $\forall s \in \mathcal{S}\forall a \in \mathcal{A}(s),\pi(a|s)>0$, 但会逐渐逼近确定的$\pi_*$ 。  
- first-visit MC 算法的ε-soft版本：  
1. 初始化:  
   $\forall s\in\mathcal{S}\forall a\in\mathcal{A}(s)$:  
   - $Q(s,a)\leftarrow$ 任意值  
   - $Returns(s,a)\leftarrow$ 空表  
   - $\pi(a,s)\leftarrow$ 任意的$\epsilon$-soft 策略(例如$\epsilon$-greedy策略)
2. 循环`while True`:  
   - 利用策略$\pi$进行一次实验  
   - `for (s,a) in`实验中出现的所有状态-动作对  
      - $G \leftarrow$ 首次出现(s,a)对 之后得到的回报
      - $Returns(s,a)$`.append`($G$)  
      - $Q(s,a)\leftarrow$`np.mean`($Returns(s,a)$)  
   - `for s in`实验中出现的所有状态  
      - $a^*\leftarrow \underset{a}{\argmax}\ Q(s,a)$  
      - $\pi(a,s)\leftarrow \begin{cases}
          1-\epsilon+\epsilon/|\mathcal{A}(s)|,&a=a^*\\ 
          \epsilon/|\mathcal{A}(s)|,&a\not ={a^*}
      \end{cases}\forall a \in \mathcal{A}(s)$

> 证明：For any ε-soft policy, $π$, any ε-greedy policy with respect to $q_π$ is guaranteed to be better than or equal to $π$.（On-Policy 策略改进定理）

证明如下：  
令$\pi'(s)$为$\epsilon$-greedy策略。  
$$
\begin{aligned}
    q_\pi(s,\pi'(s)) &= \sum_a\pi'(a|s)q_\pi(s,a)\\ &= \frac{\epsilon}{|\mathcal{A}(s)|}\sum_aq_\pi(s,a)+(1-\epsilon)\underset{a}{\max}\ q_\pi(s,a)\\ &\geq \frac{\epsilon}{|\mathcal{A}(s)|}\sum_aq_\pi(s,a)+(1-\epsilon)\sum_a\frac{\pi(a|s)-\epsilon/|\mathcal{A}(s)|}{1-\epsilon}q_\pi(s,a)\\ &= \sum_a\pi(a|s)q_\pi(s,a)=v_\pi(s)
\end{aligned}
$$

### 5.5 Off-Policy Predicts  
> off-policy methods: 要估计的策略是$\pi$，但进行实验的时候是服从策略$\mu$，$\mu\not ={\pi}$。其中，$\pi$被称为*target policy*,而$\mu$被称为*behavior policy*。目标要从服从策略$\mu$的这些实验来估计服从策略$\pi$的值。
> 重要性采样的作用：利用从某种分布的采样数据估计服从另一种分布的期望值。

#### 5.5.1 普通重要性采样  
定义重要性采样比率：  
$$
\rho_t^T=\prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{\mu(A_k|S_k)}
$$

这也是target policy和behavior policy下从时间t到终止时间T的状态-动作对的相对转移概率。  
用$\mathcal{T}(s)$代表第一次访问状态s的实验所含时刻的集合；用$T(t)$代表时刻t所在实验的终止时间，即从t开始第一次结束的时刻。$G_t$代表从时刻t到$T(t)$的回报。于是，可以得出普通重要性采样估计$v_\pi(s)$的公式：  
$$
V(s)=\frac{\sum_{t\in\mathcal{T}(s)}\rho_t^{T(t)}G_t}{\vert\mathcal{T}(s)\vert}
$$

#### 5.5.2 加权重要性采样  
有时，如果采样分布对$\pi$期望估计偏差较大时，普通的重要性采样就不满足无偏估计。于是引入加权重要性采样：  
$$
V(s)=\frac{\sum_{t\in\mathcal{T}(s)}\rho_t^{T(t)}G_t}{\sum_{t\in\mathcal{T}(s)}\rho_t^{T(t)}}
$$

#### 5.5.3 增量式的蒙特卡罗算法实现  
