# EFGCL: Learning Dynamic Motion through Spotting-Inspired External Force Guided Curriculum Learning

Keita Yoneda<sup>1</sup> , Kento Kawaharazuka1,<sup>2</sup> , Kei Okada<sup>1</sup>

*Abstract*— Learning dynamic whole-body motions for legged robots through reinforcement learning (RL) remains challenging due to the high risk of failure, which makes efficient exploration difficult and often leads to unstable learning.

In this paper, we propose External Force Guided Curriculum Learning (EFGCL), a guided RL approach based on the principle of *physical guidance*, in which external assistive forces are introduced during training. Inspired by spotting in artistic gymnastics, EFGCL enables agents to physically experience successful motion executions without relying on task-specific reward shaping or reference trajectories.

Experiments on a quadrupedal robot performing Jump, Backflip, and Lateral-Flip tasks demonstrate that EFGCL accelerates learning of the Jump task by approximately a factor of two and enables the acquisition of complex wholebody motions that conventional RL methods fail to learn. We further show that the learned policies can be deployed on a real robot, reproducing motions consistent with those observed in simulation.

These results indicate that physically guided exploration, which allows agents to experience success early in training, is an effective and general strategy for improving learning efficiency in dynamic whole-body motion tasks.

# I. INTRODUCTION

To achieve high locomotion performance in unstructured environments, quadrupedal robots require learning methods that can stably acquire a wide range of diverse and complex motor skills. With recent advances in reinforcement learning (RL), numerous approaches have been proposed that enable robust learning of individual behaviors, such as locomotion over rough terrain [1]–[5].

For tasks such as rough-terrain locomotion and obstacle traversal, learning methods with high success rates have already been established. In contrast, learning dynamic motor skills involving high acceleration and high energy, as exemplified by sports motions [6], still requires substantial taskspecific tuning. This difficulty arises because motions with a high risk of failure are inherently difficult to explore, and learning rarely progresses without explicit guidance.

To address the learning of such dynamic motions, Guided Reinforcement Learning (Guided-RL), which introduces assistance during the learning process, has been widely studied [7]. Representative approaches include imitation learning based on reference trajectories and reward shaping, which guides behavior through carefully designed reward functions.

![](_page_0_Picture_13.jpeg)

Fig. 1. Conceptual overview of External Force Guided Curriculum Learning (EFGCL). By applying external assistive forces in the early stages of learning, the agent experiences motion sequences with a higher probability of success. As learning progresses, the assistance is gradually reduced in a curriculum manner, ultimately enabling the agent to acquire a policy that achieves the target motion without assistance.

<span id="page-0-0"></span>Imitation learning directly mimics reference trajectories that represent target motions and has been increasingly applied to dynamic tasks [8]–[10]. However, the performance of the learned motions strongly depends on the quality of the reference trajectories. Methods such as Opt-Mimic [11], which generate trajectories through optimization, can provide high-quality references, but they incur substantial costs in robot modeling and objective function design. Alternatively, approaches such as WASABI [12] utilize demonstrations obtained by physically guiding the robot, reducing data collection costs. However, the quality of such data depends heavily on the skill of the human operator, making it difficult to ensure stability. As a result, there exists an inherent tradeoff between the quality of reference trajectories and the cost of generating them, and achieving high performance solely through imitation learning remains expensive.

Reward shaping aims to facilitate exploration by designing intermediate rewards that capture key elements of the desired behavior [13]–[15]. However, determining which aspects of a motion should be defined as intermediate rewards is highly task-dependent and non-trivial. Moreover, intermediate rewards may introduce designer bias, which can exclude potentially optimal motion sequences and degrade learning performance [16]. For these reasons, [16] recommends using sparse reward functions and promoting learning through

<sup>1</sup> The authors are with the Department of Mechano-Informatics, Graduate School of Information Science and Technology, The University of Tokyo, 7- 3-1 Hongo, Bunkyo-ku, Tokyo, 113-8656, Japan. [yoneda, kawaharazuka, k-okada]@jsk.imi.i.u-tokyo.ac.jp

<sup>2</sup> The author is with the AI Center, Graduate School of Information Science and Technology, The University of Tokyo, Japan.

design choices outside the reward itself. Nevertheless, a general framework for efficiently exploring dynamic motions with a high risk of failure has yet to be established.

Insightful inspiration can be drawn from artistic gymnastics, where dynamic motions are the primary objective. In gymnastics training, a technique known as *spotting* is commonly used, in which a coach physically supports the athlete while practicing a skill [17]. This approach assists the exploration process through physical guidance and differs fundamentally from conventional methods that guide behavior solely through reward design.

Motivated by this idea, we propose *External Force Guided Curriculum Learning* (EFGCL), which applies this principle to reinforcement learning for robots (Fig. 1). The main contributions of this work are summarized as follows:

- We introduce a new learning paradigm for dynamic motor skill acquisition that employs *physical guidance* via external forces, rather than guidance through reward design.
- We propose External Force Guided Curriculum Learning (EFGCL), which gradually decays external assistive forces and demonstrate that physical guidance significantly improves exploration efficiency.
- Through learning experiments on a real quadrupedal robot, we demonstrate that the proposed approach is effective and transferable to real-world environments.

#### II. BACKGROUND

#### A. Proximal Policy Optimization (PPO)

In reinforcement learning for legged robots, Proximal Policy Optimization (PPO) [18] is commonly used due to its training stability and ease of implementation. PPO constrains the magnitude of policy updates by optimizing a clipped surrogate objective function based on the likelihood ratio between the current and previous policies,

$$r_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}, \quad a_t \sim \pi_{\theta_{\text{old}}}(\cdot \mid s_t),$$

thereby enabling stable learning while limiting excessive policy updates. This update control is particularly effective for legged robots with high-dimensional action spaces and unstable dynamics, as it prevents training collapse caused by abrupt policy changes.

PPO also employs Generalized Advantage Estimation (GAE) [19] to estimate the advantage  $A_t$ . GAE computes the advantage as a weighted sum of temporal-difference (TD) errors,  $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ , where  $\gamma \in [0,1]$  is the discount factor. This formulation allows a trade-off between variance and bias. Based on the estimated  $A_t$ , PPO updates the policy in the following gradient direction:

$$\nabla_{\theta} J(\theta) \propto \mathbb{E}_{s_t, a_t \sim \pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta} (a_t \mid s_t) A_t \right].$$

However, in environments with a high risk of failure, unsuccessful trajectories tend to dominate, resulting in small estimated state values  $V(s_t)$  for many states. Consequently, both the TD error  $\delta_t$  and the advantage  $A_t$  approach zero, providing little useful information for gradient-based updates.

# <span id="page-1-0"></span>Algorithm 1 External Force Guided Curriculum Learning

- 1: **Input:** initial policy  $\pi_{\text{init}}$ , initial critic  $V_{\text{init}}$ , assist force  $F_{\text{assist}}$ , decay step size  $\varepsilon$ , success threshold  $\zeta$
- 2: **Output:** final policy  $\pi_N$
- 3: Initialize:

$$F_0 \leftarrow F_{\text{assist}}, \quad \pi_0 \leftarrow \pi_{\text{init}}, \quad V_0 \leftarrow V_{\text{init}}$$

4: Set decay rate:

$$\alpha \leftarrow 1.0$$

- 5: **for** i = 0, ..., N-1 **do**
- 6:  $success\_rate \leftarrow 0$
- 7: **while**  $success\_rate < \zeta$  **do**
- 8: Train  $\pi_i$  and  $V_i$  using PPO under assist force  $F_i$
- 9: Update *success\_rate*
- 10: end while
- 11: Decay assist force:

$$\alpha \leftarrow \max(0, 1 - \varepsilon \times i), \quad F_{i+1} \leftarrow \alpha \times F_{\text{assist}}$$

12: Carry over the learned policy and value function:

$$\pi_{i+1} \leftarrow \pi_i, \quad V_{i+1} \leftarrow V_i$$

- 13: end for
- 14: **return**  $\pi_N$

Thus, while PPO is a stable optimization method, learning can stagnate severely in environments where successful experiences are rarely obtained.

#### <span id="page-1-2"></span>B. Curriculum Learning

Curriculum learning has been widely adopted as a representative approach to address the aforementioned issue.

In curriculum learning, training begins in specialized environments with low exploration risk, and the task difficulty or risk level is gradually increased. This framework can be interpreted as sequential learning over multiple Markov Decision Processes (MDPs)  $\mathcal{M}_i$  with different risk levels. If the changes in dynamics and reward structures between successive stages are sufficiently small, the optimal policy  $\pi_i^*$  for stage  $\mathcal{M}_i$  and the optimal policy  $\pi_{i+1}^*$  for the subsequent stage  $\mathcal{M}_{i+1}$  are expected to be similar. That is,

$$\frac{\pi_{i+1}^*(a_t \mid s_t)}{\pi_i^*(a_t \mid s_t)} \approx 1, \quad a_t \sim \pi_i^*(\cdot \mid s_t),$$

can be artificially constructed. This property is consistent with the assumption underlying PPO that smaller policy update steps are preferable. Therefore, curriculum learning provides a theoretically well-aligned framework that supports stable, incremental learning with PPO.

# <span id="page-1-1"></span>C. Stabilizing Learning via Accelerated Value Function Estimation

In reinforcement learning with sparse rewards, learning efficiency strongly depends on how quickly the value function can correctly evaluate action sequences that yield high rewards. This is particularly critical for methods such as

![](_page_2_Figure_0.jpeg)

<span id="page-2-1"></span>Fig. 2. Network architecture used for Teacher–Student learning. (a) The Teacher Policy takes the full state, including privileged observations, as input and is trained via reinforcement learning. (b) The Student Policy takes only onboard sensor information as input and is trained using the Teacher Policy as a supervisor.

PPO, which rely on value functions for advantage estimation, where the initial accuracy of value estimation significantly influences the direction of policy updates.

In general, once a sufficient number of high-reward trajectories are observed, the state values V(s) corresponding to the associated state sequence  $\xi = \{s_t^{\text{high}}\}_{t=0}^N$  are estimated to be large. When such high-value trajectories exist, actions that deviate from them yield large negative TD errors and advantages. As a result, the policy is updated in a direction that discourages deviation from high-reward trajectories, making effective motion sequences more likely to be preserved once acquired.

Therefore, whether the value function can assign high values to near-optimal behaviors from the early stages of learning is a crucial factor that determines the overall stability and efficiency of training. To achieve this, it is effective to expose the agent to a large number of high-reward trajectories during the early phase of learning.

#### III. METHOD

# A. Overview of External Force Guided Curriculum Learning (EFGCL)

The proposed External Force Guided Curriculum Learning (EFGCL) provides a framework for achieving stable learning in dynamic tasks while maintaining sparse reward functions. Instead of modifying rewards as in imitation learning or Reward Shaping, EFGCL stabilizes learning by curriculumwise modifying the Markov Decision Process (MDP) itself in which learning is performed.

The overall procedure of the algorithm is summarized in Alg. 1. EFGCL designs external assistive forces that facilitate task execution during the early stage of learning (line 1), and gradually decays this assistance as training progresses (lines 7–9). Through this curriculum process, the agent experiences trajectories with a high probability of success under assistance, while gradually transitioning toward autonomous motion generation. Eventually, the agent acquires a policy  $\pi_N$  that achieves the target motion without assistance (line 14).

![](_page_2_Picture_9.jpeg)

Fig. 3. Overview and kinematic structure of the quadrupedal robot KLEIYN. (a) Overall robot structure, (b) link and joint definitions for each leg.

#### <span id="page-2-0"></span>B. Design of External Assistance

EFGCL first designs external assistive forces that help reproduce the target motions. As discussed in Sec. II-C, exposing the agent to a large number of high-reward trajectories in the early stage of learning induces a tendency to preserve such trajectories. The external assistance in EFGCL is introduced to artificially increase the density of these successful trajectories.

In EFGCL, external assistance is defined as a pattern consisting of three elements: the points of application  $P = \{\mathbf{p}_i\}$ , the corresponding force vectors  $F = \{\mathbf{f}_i\}$ , and the timing of application  $T = \{(t_i^{\text{start}}, t_i^{\text{end}})\}$ . Since the purpose of the assistance is not to teach an optimal trajectory but rather to guide the agent toward high-reward states, it is sufficient for the robot to approximately achieve the target motion and obtain high rewards. Therefore, in this study, the assistive force  $F_{\text{assist}}(P, F, T)$  is heuristically designed for each task. The permissible range of such assistive forces is investigated in detail in Sec. V-B.

#### C. Success-Rate Based Adaptive Curriculum Scheduling

As discussed in Sec. II-B, curriculum learning benefits from small difficulty gaps between adjacent MDPs to ensure stable policy transitions. EFGCL adopts a success-rate-based adaptive curriculum to prevent excessive difficulty changes caused by curriculum updates.

Specifically, as shown in lines 6–12 of Alg. 1, PPO training is repeated at each stage i until the success rate exceeds a threshold  $\zeta$ . The assistive force is then updated as

$$F_i = \alpha_i \times F_{\text{assist}}, \quad \alpha_i = \max(0, 1 - \varepsilon \times i).$$

where  $\alpha_i$  denotes the assistance scaling factor at stage i, and  $\varepsilon$  is the decay step size controlling the rate of assistance reduction.

This mechanism enables automatic adjustment of the assistance decay step based on training progress, while preserving the relationship  $\pi_{i+1}(a \mid s)/\pi_i(a \mid s) \approx 1$ .

#### D. Time-Encoding for Observations

Since the assistive force  $F_{\rm assist}(P,F,T)$  in EFGCL is applied in a time-dependent manner, it is important for the agent to infer the timing of assistance as an internal state.

However, directly using the elapsed time t as an input leads to monotonically increasing values, which may cause scale mismatch in neural networks.

To address this issue, we introduce an additional observation that aligns with the activation interval of the assistive force  $F_i$ . Specifically, we define a monotonically increasing function bounded within [0,1] as

$$\tau(t,\lambda) = \frac{\tilde{t}^3}{1+\tilde{t}^3}, \quad \tilde{t} = \frac{t}{\lambda}.$$

Here,  $\lambda$  is a temporal scaling parameter, which is set to the force activation start time  $\lambda = t^{\text{start}}$  in this study.

![](_page_3_Figure_3.jpeg)

<span id="page-3-0"></span>Fig. 4. Designed assistive force patterns for each task. (a) Jump, (b) Backflip, (c) Lateral-flip. The assistive forces are applied vertically to the scapula links, guiding motions that facilitate successful task execution during the early stages of learning.

#### IV. LEARNING SETUP

#### A. Robot Platform

The real-world experiments are conducted using the quadrupedal robot KLEIYN [6]. Its appearance and link definitions are shown in Fig. 3. KLEIYN has a total mass of 18 kg and a height of 600 mm, with three degrees of freedom (DoF) per leg and one DoF in the torso. The robot is equipped with an IMU and joint encoders. The leg motors are quasi-direct-drive actuators with a maximum torque of 24.8 Nm, while the torso motor has a maximum torque of 48 Nm. Isaac Gym [20] is used as the simulator for training.

#### B. Task Definition and Reward Function

To evaluate the effectiveness of EFGCL, we define three dynamic whole-body motion tasks: (1) Jump, (2) Backflip, and (3) Lateral-flip.

To avoid arbitrary performance gains due to task-specific reward engineering, all tasks share exactly the same reward structure, weights, and functional forms. Only the target variables differ between tasks: the maximum height for Jump, and the rotation angle for Backflip and Lateral-flip. These target variables are simply scaled according to their physical units, and no task-specific reward tuning or intermediate motion-guiding rewards are introduced.

The reward for each task is defined by the following common structure:

$$r_t = \rho_t^{\text{task}} + \rho_t^{\text{task}} \cdot \rho_t^{\text{stand}} + \lambda_{\omega} r_t^{\text{ang}} + r_t^{\text{common}}, \qquad (1)$$

where  $\rho_t^{\mathrm{task}} \in [0,1]$  represents task progress,  $\rho_t^{\mathrm{stand}}$  encourages stable posture after landing,  $r_t^{\mathrm{ang}}$  is a regularization term that suppresses rotation about non-target axes, and  $r_t^{\mathrm{common}}$  enforces physical constraints shared across all tasks.

The only task-specific difference lies in the definition of the target quantity in  $\rho_t^{\text{task}}$ . For Jump, the target is the maximum achieved height, while for Backflip and Lateral-flip, the targets are the rotation angles around the pitch and roll axes, respectively. Detailed definitions are summarized in Appendix I.

#### C. Observations

The observation consists of two components,  $\mathbf{o}_t^{\text{prop}}$  and  $\mathbf{o}_t^{\text{priv}}$

 $\mathbf{o}_t^{\mathrm{prop}}$  represents proprioceptive observations and includes joint positions  $\mathbf{q}_t \in \mathbb{R}^{13}$ , joint velocities  $\dot{\mathbf{q}}_t \in \mathbb{R}^{13}$ , the gravity vector in the root frame  $\tilde{\mathbf{g}}_t \in \mathbb{R}^3$ , root angular velocity  $\omega_t \in \mathbb{R}^3$ , the command input  $c \in \mathbb{R}$ , and the time encoding  $\tau(t) \in \mathbb{R}$ . The command input corresponds to the target jump height  $h^{\mathrm{target}}$  for Jump, and the target rotation angle  $\theta^{\mathrm{target}}$  for Backflip and Lateral-flip.

 $\mathbf{o}_t^{\text{priv}}$  denotes privileged observations used exclusively by the Teacher Policy and consists of the following task-specific information:

- Jump: root height  $h_t \in \mathbb{R}$  and maximum height since episode start  $h_t^{\max} \in \mathbb{R}$
- Backflip: root height  $h_t$  and root pitch angle  $\theta_t^{\text{pitch}} \in \mathbb{R}$
- Lateral-flip: root height  $h_t$  and root roll angle  $\theta_t^{\text{roll}} \in \mathbb{R}$

#### D. Assist Force Design

Assistive force patterns are designed for each task. The application timing is shared across all tasks as  $T = \{(1.0s, 1.1s)\}$ , while the application points P and force vectors F are task-specific. The assistive force patterns are illustrated in Fig. 4.

a) Jump:

$$\begin{split} P^{\text{jump}} &= \{\mathbf{p}_{\text{FL}}^{\text{Scapula}}, \mathbf{p}_{\text{FR}}^{\text{Scapula}}, \mathbf{p}_{\text{BL}}^{\text{Scapula}}, \mathbf{p}_{\text{BR}}^{\text{Scapula}}\}, \\ F^{\text{jump}} &= \{(0, 0, f_{\text{jump}}(h^{\text{target}})/4)\} \end{split}$$

Here,  $f_{\text{jump}}(h^{\text{target}})$  is the assistive force magnitude determined by the target height  $h^{\text{target}}$ . This value is derived by modeling the Jump motion as simple projectile motion and computing the average assistive force required to generate the initial velocity needed to reach the target height.

b) Backflip:

$$\begin{split} P^{\text{backflip}} &= \{\mathbf{p}_{\text{FL}}^{\text{Scapula}}, \mathbf{p}_{\text{FR}}^{\text{Scapula}}\}, \\ F^{\text{backflip}} &= \{(0, 0, 175\,\text{N})\} \end{split}$$

*c) Lateral-flip:*

$$\begin{split} \textit{P}^{lateral} &= \{ \boldsymbol{p}_{FR}^{Scapula}, \boldsymbol{p}_{BR}^{Scapula} \}, \\ \textit{F}^{lateral} &= \{ (0,0,300\,\mathrm{N}) \} \end{split}$$

### E. Adaptive Curriculum Design

A success-rate-based curriculum scheduling strategy is employed to decay the assistive forces. The success rate in Alg. 1 is computed using the following criteria.

a) Jump:

$$|h_t^{\text{max}} - h^{\text{target}}| < 0.1 \land |h_t| < 0.1$$

b) Backflip, Lateral-flip:

$$|\theta_t - 2\pi| < 0.3 \wedge |h_t| < 0.1$$

The success rate threshold is set to  $\zeta = 0.6$ , and the decay step size is set to  $\varepsilon = 0.01$  for all tasks.

![](_page_4_Figure_0.jpeg)

<span id="page-4-1"></span>Fig. 5. Comparison of learning curves with and without EFGCL. (a) Jump, (b) Backflip, (c) Lateral-flip. With EFGCL, learning progresses stably and converges to high reward values in all tasks. Without EFGCL, Jump exhibits large reward variance, while learning hardly progresses for Backflip and Lateral-flip.

# *F. Teacher–Student Learning*

Markovianity is a crucial property in reinforcement learning environments. Following prior work [2], we adopt a Teacher–Student learning architecture.

Teacher–Student learning consists of two stages: reinforcement learning of the teacher policy and supervised learning of the student policy, where the teacher policy acts as a supervisor. During supervised learning, the student policy is trained to minimize an action-matching loss and a reconstruction loss on privileged observations. Since the proposed method is integrated with reinforcement learning, EFGCL is applied only during the training of the teacher policy. The overall learning structure is illustrated in Fig. [2.](#page-2-1)

# V. EXPERIMENT AND RESULT

#### *A. Learning Performance and Real Robot Deployment*

We compared the proposed EFGCL with a PPO baseline over 10 random seeds. As shown in Fig. [5,](#page-4-1) EFGCL achieved stable convergence and high rewards across all tasks (Jump, Backflip, and Lateral-flip). In contrast, the baseline failed to learn the flipping tasks and exhibited high variance in the Jump task. As visualized in Fig. [6,](#page-4-2) the baseline often resulted in unnatural postures, whereas EFGCL acquired natural dynamic motions. This stable learning process was facilitated by the adaptive curriculum, which automatically adjusted the assistive force decay based on the success rate (Fig. [7\)](#page-4-3). Furthermore, the policies learned via EFGCL were distilled and deployed on the quadrupedal robot KLEIYN. As shown in Fig. [8,](#page-5-0) the dynamic motions observed in simulation were successfully reproduced on the real robot for all three tasks.

#### <span id="page-4-0"></span>*B. Ablation Study of EFGCL Force Design*

To evaluate the sensitivity of EFGCL to heuristic design choices, we varied the application point, magnitude, and timing of the assistive force in the Backflip task. The results in Fig. [9](#page-5-1) demonstrate that learning is robust over a wide range of parameters. Successful policies were acquired even when the force was applied to different links (thigh or calf) or when the magnitude varied within a reasonable range (140–210 N). Learning failed only in extreme cases where the assistance was physically insufficient (e.g., 100 N) or excessive (e.g., 250 N), or when the application timing was too short (1.0 s

![](_page_4_Figure_10.jpeg)

<span id="page-4-2"></span>Fig. 6. Snapshots of learned motions with and without EFGCL. (a–c) With EFGCL, (d–f) without EFGCL. With EFGCL, natural and stable motions are acquired, whereas without EFGCL, unnatural postures and failure to achieve the target motions are observed.

![](_page_4_Figure_12.jpeg)

<span id="page-4-3"></span>Fig. 7. Transitions of the success rate and assistive force decay factor in EFGCL. (a) Jump task, (b) Backflip and Lateral-flip tasks. The decay speed of the assistive force is automatically adjusted according to the success rate.

![](_page_5_Picture_0.jpeg)

Fig. 8. Reproduction of learned motions on the real quadrupedal robot. (a) Jump, (b) Backflip, (c) Lateral-flip.

<span id="page-5-0"></span>![](_page_5_Figure_2.jpeg)

<span id="page-5-1"></span>Fig. 9. Ablation study on assistive force design. (a) Application points, (b) force magnitudes, (c) application timing. The asterisks (\*) in the legends indicate the conditions used in the main experiments.

to 1.05 s). These results indicate that precise tuning is not required, as long as the assistance roughly facilitates the target motion.

# *C. Evaluation of Accelerated Critic Value Estimation*

To validate the hypothesis that external guidance accelerates critic learning, we analyzed value estimates during the Jump task. Fig. [10](#page-5-2) compares the value function outputs for a successful reference motion at different training stages. With EFGCL, the value estimates converged to the final distribution as early as 200 iterations. In contrast, the baseline required more than 1,000 iterations to reach a comparable

![](_page_5_Figure_7.jpeg)

<span id="page-5-2"></span>Fig. 10. Comparison of value function estimation with and without EFGCL. With EFGCL, value estimates quickly converge to distributions close to the final value function, demonstrating accelerated and stabilized value estimation.

level of accuracy and exhibited larger variance. This result confirms that experiencing successful states early in training significantly accelerates value function estimation.

#### VI. DISCUSSION

#### *A. Efficacy and Robustness of Guided Exploration*

The experimental results demonstrate that EFGCL significantly stabilizes the learning of dynamic motion skills by accelerating value function estimation during the early training phase. Unlike reward shaping or imitation learning, which rely on complex reward design or expert datasets, EFGCL guides exploration through direct physical assistance in the form of external forces.

Furthermore, the ablation study shows that the proposed method is highly robust to variations in assistive force design. As long as the assistance roughly facilitates the target motion, learning can succeed without precise parameter tuning. These results suggest that the principle of "physically experiencing success" provides a general and cost-effective strategy for overcoming exploration challenges in dynamic robotic reinforcement learning.

# *B. Limitations and Future Work*

This study focuses on validating the principle of artificially enabling agents to experience successful motions. Accordingly, the assistive forces were designed based on task-specific physical intuition. While such heuristic designs are sufficient for the single-shot dynamic motion tasks considered in this work, the design burden may increase for more complex and continuous motions.

For continuous behaviors such as dancing, maintaining the overall motion structure often requires learning based on reference trajectories. In such cases, using the proposed framework of external forces with sparse rewards alone may be insufficient, and combining it with trajectory-tracking reward designs or imitation learning is likely to be more effective. Developing automatic optimization or generation methods for assistive forces that remain effective for complex target motions is an important direction for future work.

#### VII. CONCLUSION

Inspired by spotting in gymnastics, we proposed External Force Guided Curriculum Learning (EFGCL), a reinforcement learning framework that guides exploration through decaying external forces. Without relying on complex reward shaping or reference trajectories, EFGCL enables a quadrupedal robot to acquire dynamic whole-body motions, such as jumping and flipping, that are difficult for standard RL methods.

Through both simulation and real-robot experiments, we demonstrated successful sim-to-real transfer and showed that physical assistance accelerates value function estimation by allowing the agent to experience successful states early in training. Although the current approach relies on heuristic force design, the results suggest that *physical guidance* represents a promising and general paradigm for guided exploration, complementary to reward-based and imitation-based methods, in the learning of complex whole-body motions.

# <span id="page-6-0"></span>APPENDIX I REWARD DEFINITIONS

Table I summarizes the reward terms shared across all tasks and their definitions. Here,  $P_{col}$  denotes the set of link indices used to detect collisions with the ground. This set includes 14 links in total: the body, scapula, thigh, and calf, excluding the feet. The indicator function  $\delta_t^{\text{term}}$  takes the value of 1 if the episode terminates due to the trunk contacting the ground at time t, and 0 otherwise.

# A. Task-Specific Target Variables

Table II presents the task-specific definitions of the target variable  $x_t$ , the target value  $x^{\text{target}}$ , and the normalization coefficient  $s_x$  used in the task progress reward  $\rho_t^{\text{task}}$  shown in Table I.

<span id="page-6-1"></span>TABLE I
SUMMARY OF REWARD TERMS SHARED ACROSS ALL TASKS.

| Reward term                | Definition                                                                                                                                           |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| Task progress              | $\rho_t^{\text{task}} = \exp(-\ x_t - x^{\text{target}}\ ^2 / s_x)$                                                                                  |
| Standing                   | $ \rho_t^{\text{stand}} = \exp\left(-\frac{\ h_t\ ^2}{0.01}\right) + \exp\left(-\frac{\ \mathbf{q}_t - \mathbf{q}^{\text{stand}}\ ^2}{0.25}\right) $ |
| Angular regularization     | $r_t^{\rm ang} = -\ (\boldsymbol{\omega}_t^{\rm non-target})^2\ $                                                                                    |
| Collision penalty          | $-1.0 \times \sum_{i \in P_{col}} (f_{i,z} > 0.1)$                                                                                                   |
| Termination penalty        | $-100 \times \delta_t^{\text{term}}$                                                                                                                 |
| Joint velocity penalty     | $-5 \times 10^{-4} \ \dot{\mathbf{q}}_t\ ^2$                                                                                                         |
| Joint acceleration penalty | $-1\times10^{-7}\ \ddot{\mathbf{q}}_t\ ^2$                                                                                                           |

#### REFERENCES

- J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter, "Learning agile and dynamic motor skills for legged robots," *Science Robotics*, vol. 4, no. 26, 2019.
- [2] T. Miki, J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, "Learning robust perceptive locomotion for quadrupedal robots in the wild," *Science Robotics*, vol. 7, no. 62, 2022.

<span id="page-6-2"></span>TABLE II
TASK-SPECIFIC INSTANTIATIONS OF THE TASK PROGRESS REWARD.

| Task         | Target variable $x_t$     | x <sup>target</sup> | $S_X$   |
|--------------|---------------------------|---------------------|---------|
| Jump         | $h_t^{\max}$              | htarget             | 0.01    |
| Backflip     | $\theta_t^{\text{pitch}}$ | $2\pi$              | $\pi^2$ |
| Lateral-Flip | $\theta_t^{\rm roll}$     | $2\pi$              | $\pi^2$ |

- [3] Z. Zhuang, Z. Fu, J. Wang, C. Atkeson, S. Schwertfeger, C. Finn, and H. Zhao, "Robot parkour learning," in *Conference on Robot Learning* (CoRL), 2023.
- [4] D. Vogel, R. Baines, J. Church, J. Lotzer, K. Werner, and M. Hutter, "Robust ladder climbing with a quadrupedal robot," in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025, pp. 7239–7244.
- [5] H. Kim, H. Oh, J. Park, Y. Kim, D. Youm, M. Jung, M. Lee, and J. Hwangbo, "High-speed control and navigation for quadrupedal robots on complex and discrete terrain," *Science Robotics*, vol. 10, no. 102, p. eads6192, 2025.
- [6] K. Yoneda, K. Kawaharazuka, T. Suzuki, T. Hattori, and K. Okada, "Kleiyn: A quadruped robot with an active waist for both locomotion and wall climbing," in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025, pp. 8783–8789.
- [7] J. Eßer, N. Bach, C. Jestel, O. Urbann, and S. Kerner, "Guided reinforcement learning: A review and evaluation for efficient and effective real-world robotics [survey]," *IEEE Robotics & Automation Magazine*, vol. 30, no. 2, pp. 67–85, 2022.
- [8] X. B. Peng, P. Abbeel, S. Levine, and M. Van de Panne, "Deepmimic: Example-guided deep reinforcement learning of physics-based character skills," ACM Transactions On Graphics (TOG), vol. 37, no. 4, pp. 1–14, 2018.
- [9] X. B. Peng, E. Coumans, T. Zhang, T.-W. E. Lee, J. Tan, and S. Levine, "Learning agile robotic locomotion skills by imitating animals," in *Robotics: Science and Systems*, 07 2020.
- [10] J. Wu, G. Xin, C. Qi, and Y. Xue, "Learning robust and agile legged locomotion using adversarial motion priors," *IEEE Robotics* and Automation Letters, vol. 8, no. 8, pp. 4975–4982, 2023.
- [11] Y. Fuchioka, Z. Xie, and M. Van de Panne, "Opt-mimic: Imitation of optimized trajectories for dynamic quadruped behaviors," in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 5092–5098.
- [12] C. Li, M. Vlastelica, S. Blaes, J. Frey, F. Grimminger, and G. Martius, "Learning agile skills via adversarial imitation of rough partial demonstrations," in *Conference on Robot Learning*. PMLR, 2023, pp. 342–352.
- [13] S. Devlin and D. Kudenko, "Theoretical considerations of potential-based reward shaping for multi-agent systems," in *Tenth international conference on autonomous agents and multi-agent systems*. ACM, 2011, pp. 225–232.
- [14] V. Atanassov, J. Ding, J. Kober, I. Havoutis, and C. Della Santina, "Curriculum-based reinforcement learning for quadrupedal jumping: A reference-free design," *IEEE Robotics & Automation Magazine*, 2024.
- [15] G. Bellegarda, C. Nguyen, and Q. Nguyen, "Robust quadruped jumping via deep reinforcement learning," *Robotics and Autonomous Systems*, vol. 182, p. 104799, 2024.
- [16] G. Vasan, Y. Wang, F. Shahriar, J. Bergstra, M. Jägersand, and A. R. Mahmood, "Revisiting sparse rewards for goal-reaching reinforcement learning," *Reinforcement Learning Journal*, vol. 4, pp. 1841–1854, 2024.
- [17] S. Sorzano, "How spotting with touch affects skill performance and self confidence in gymnasts," Master's thesis, Trent University (Canada), 2023.
- [18] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," 2017.
- [19] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, "High-dimensional continuous control using generalized advantage estimation," in *International Conference on Learning Representations* (ICLR), 2016.
- [20] J. Liang, V. Makoviychuk, A. Handa, N. Chentanez, M. Macklin, and D. Fox, "Gpu-accelerated robotic simulation for distributed reinforcement learning," in *Conference on Robot Learning (CoRL)*, 2018.

---

## Notes

- **Title:** EFGCL: Learning Dynamic Motion through Spotting-Inspired External Force Guided Curriculum Learning
- **URL:** https://arxiv.org/pdf/2605.10063
