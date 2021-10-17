## Chapter 4: Dynamic Programming  
### 4.1 Policy Evaluation  
> $\forall s\in \mathcal{S}$:  
> $$
\begin{aligned}
    v_\pi(s)&=\mathrm{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]\\ 
    &=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]
\end{aligned}
> $$  

迭代策略评估(*iterative policy evaluation*)算法的依据是如下的贝尔曼方程：  
$$
\begin{aligned}
    v_{k+1}(s)&=\mathrm{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]\\ 
    &=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]
\end{aligned}\tag{4.1}
$$

当$k\rightarrow \infty$，{$v_k$}会逐渐收敛于固定值$v_\pi$。这样就用迭代的方法实现了对任意状态s处采取策略$\pi$的值$v_\pi(s)$即对策略$\pi$的评估。  
策略迭代评估算法的具体步骤：  
```flow
st=>start: 开始
it=>inputoutput: π
op1=>operation: V=np.zeros([1,len(states)])
op=>operation: update V, ∆
e=>end
cond=>condition: ∆ < θ?
ot=>inputoutput: V
st(right)->it->op1->op->cond
cond(no)->op
cond(yes)->ot(right)->e
```  
其中 "update V, $\Delta$"的流程如下：  
$\Delta\leftarrow0$  
`for s in states`:    
- $v\leftarrow V(s)$  
- $V(s)\leftarrow \mathrm{equation}(4.1)$  
- $\Delta \leftarrow \max(\Delta,|v-V(s)|)$  

### 4.2 Policy Improvement  
行为值函数如果使用下一所有可能状态的值函数来表示：
$$q_\pi(s,a)=\sum_{s',r}p(s',r|s,a)(r+\gamma v_\pi(s'))\tag{4.2}$$

这就是对于在状态s处，采取动作a，后继采用策略$\pi$的行为值函数的另一种表示。  
于是，对于确定性的策略$\pi'$和策略$\pi$:  
如果$\forall s\in \mathcal{S},v_\pi(s)\leq q_\pi(s,\pi'(s))$，那么一定有$v_\pi(s)\leq v_{\pi'}(s)$  
如果$\pi'$是新的**贪婪策略**，  
$$\pi'(s)=\underset{a}{\argmax}\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]$$  

那么