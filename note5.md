## CHAMPTER 5: Monte Carlo Methods  
### 5.1 Monte Carlo Prediction  
> 蒙特卡罗方法的思想：利用在第二章的状态值方法中的思想，采用大数定律，某个状态的值是后继状态**累积的折扣回报**的期望。估计该值的方法是求得**观测**该状态对应的回报的均值，随着观测的次数的增多，均值应当收敛至状态值。这就是蒙特卡罗方法(MC method)。
> 本章聚焦于first-visit MC method，即计算首次访问状态s的回报的均值来作为s的状态值。

- first-visit MC算法:  
1. 初始化:  
$\pi \leftarrow$ 待评估的策略;  
$V \leftarrow$ 某个任意产生的状态值函数;  
$Returns(s) \leftarrow$ 空表, $\forall s\in \mathcal{S}$;  
2. 循环`while True`:  
利用$\pi$产生一次实验;  
`for s in `本次实验出现的所有状态:  
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
    - $\pi(s) \leftarrow$ arbitrary;  
    - $Q(s,a) \leftarrow$ arbitrary;  
    - $Returns(s,a) \leftarrow$ 空表,  
2. 循环`while True`:  
利用$\pi$产生一次实验，该实验从$S_0\in\mathcal{S},A_0\in\mathcal{A(\mathcal{S_0})}$开始，每个s-a对都应该符合探索性条件;  
    `for (s,a) in `本次实验出现的所有状态-行为对:  
    - $G \leftarrow$ 首次出现(s,a)对 之后得到的回报
    - $Returns(s,a)$`.append`($G$)  
    - $Q(s,a)\leftarrow$`np.mean`($Returns(s,a)$)  

   `for s in` 本实验出现的所有状态:  
    - $\pi(s)\leftarrow \underset{a}{\argmax} Q(s,a)$ (greedy)

### 5.4 On-Policy Predicts
> 探索性初始化理论上可行，但实际上却是不太可能实现的。因此，只能保证智能体继续可以选择所有的动作。
> On-Policy Predicts: $\forall s \in \mathcal{S}\forall a \in \mathcal{A}(s),\pi(a|s)>0$, 但会逐渐逼近确定的$\pi_*$ 。  
- first-visit MC 算法的ε-soft版本：  
1. 初始化:  
2. 循环`while True`:  


### 5.5 Off-Policy Predicts  
#### 5.5.1 重要性采样  

#### 5.5.2 加权重要性采样  

#### 5.5.3 增量式的蒙特卡罗算法实现  
