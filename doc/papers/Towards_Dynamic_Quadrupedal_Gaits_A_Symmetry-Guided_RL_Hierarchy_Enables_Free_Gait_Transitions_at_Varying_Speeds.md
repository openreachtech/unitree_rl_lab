# Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy Enables Free Gait Transitions at Varying Speeds

Jiayu Ding\*, Xulin Chen\*, Garrett E. Katz, and Zhenyu Gan

Abstract—Quadrupedal robots exhibit a wide range of viable gaits, but generating specific footfall sequences often requires laborious expert tuning of numerous variables, such as touchdown and lift-off events and holonomic constraints for each leg. This paper presents a unified reinforcement learning framework for generating versatile quadrupedal gaits by leveraging the intrinsic symmetries and velocity-period relationship of dynamic legged systems. We propose a symmetry-guided reward function design that incorporates temporal, morphological, and timereversal symmetries. By focusing on preserved symmetries and natural dynamics, our approach eliminates the need for predefined trajectories, enabling smooth transitions between diverse locomotion patterns such as trotting, bounding, half-bounding, and galloping. Implemented on the Unitree Go2 robot, our method demonstrates robust performance across a range of speeds in both simulations and hardware tests, significantly improving gait adaptability without extensive reward tuning or explicit foot placement control. This work provides insights into dynamic locomotion strategies and underscores the crucial role of symmetries in robotic gait design.

#### I. INTRODUCTION

Quadrupedal robots hold promise for applications such as search-and-rescue, industrial inspection, and planetary exploration. A fundamental limitation, however, is the absence of a unified framework that enables these robots to generate and switch among diverse gaits on demand. In animals, such transitions are routine: walking conserves energy, running increases speed, and galloping clears obstacles. Replicating this adaptive capability in robots would greatly expand their effectiveness in unstructured environments. Yet despite decades of research, most robotic systems remain restricted to a small set of pre-programmed gaits. Developing controllers that can flexibly coordinate leg motions across gait classes therefore remains a central challenge in legged locomotion research. Most existing approaches fall short of this goal. Trajectory optimization and model predictive control [1] rely on fixed, hand-coded footfall sequences that work well in structured conditions but degrade under uncertainty. Reinforcement learning (RL) has emerged as a powerful alternative, but current methods typically require massive hand-tuning for reward function design [2]. Central pattern generator (CPG)based controllers produce multiple gaits, but they depend on explicitly prescribed foot trajectories that may not match the robot's intrinsic dynamics.

Jiayu Ding and Zhenyu Gan are with the Department of Mechanical and Aerospace Engineering, Syracuse University, Syracuse, NY 13244 { jding14, zgan02}@syr.edu. Xulin Chen and Garrett E. Katz are with the Department of Electrical Engineering and Computer Science, Syracuse University { xchen168, gkatz01}@syr.edu. Jiayu Ding\* and Xulin Chen\* contributed equally to this publication.

This work was supported by a startup fund from the Syracuse University.

<span id="page-0-0"></span>![](_page_0_Figure_8.jpeg)

Fig. 1: Concept overview of the symmetry-guided reinforcement learning framework. User commands and gait parameters feed an MLP policy with temporal and morphological symmetries, enabling a single policy to generate trotting, bounding, half-bounding, and galloping on the Unitree Go2 without predefined trajectories.

Symmetry offers a solution to harness the advantage of both approaches. Hildebrand's taxonomy [3] classified quadrupedal gaits in terms of temporal and spatial symmetries, while later robotics studies [4], [5] showed that gaits can be derived from compositions and disruptions of temporal, spatial, and morphological symmetries. This perspective treats gaits not as isolated behaviors but as related members of broader families, and embedding symmetry into learning provides a compact representation of gait repertoires while reducing the search space for policy training.

In this paper, we present a symmetry-guided reinforcement learning framework for quadrupedal gait generation. Temporal, morphological, and time-reversal symmetries are incorporated directly into the reward function and phase mappings, guiding policy learning without predefined sequences or gait-specific tuning. Using this approach, we train a single policy that reproduces trotting, bounding, half-bounding, and galloping on the Unitree Go2 robot (Fig. 1). We evaluate the learned policy in both simulation and hardware across a wide range of commanded speeds and gait transitions, using metrics for velocity tracking accuracy, gait consistency, and energy efficiency. Results show that symmetry-enforced policies achieve more accurate velocity regulation, more coordinated footfall patterns, and reduced cost of transport compared to baselines without symmetry.

The contributions of this work are: 1) A symmetry-constrained reward design that simplifies gait learning by reducing the dimensionality of command space. 2) A unified policy that reproduces a spectrum of quadrupedal gaits on hardware, demonstrating improved tracking, consistency, and efficiency over policies trained without symmetry. By incor-

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Fig. 2: Overview of the proposed framework. Left: User commands  $(v_x^{\rm cml}, v_y^{\rm cml}, \omega_{\rm yaw}^{\rm cml})$  and selected gait sequences (trotting, bounding, half-bounding, galloping) are mapped into gait parameters  $\Gamma = [\theta_{\rm LH}, \theta_{\rm LF}, \theta_{\rm RF}, \theta_{\rm RH}, v_x^{\rm cml}]$ . Middle: Training framework design integrates reward function design (command tracking, smoothness, temporal and morphological symmetry), along with time-reversal mapping, domain randomization, and velocity/gait resampling. Right: The framework drives training of our MLP policy network, which outputs joint targets tracked by a PD controller on the Unitree Go2 robot, with a user interface providing real-time command input.

porating symmetries into reinforcement learning, this work establishes a scalable framework for adaptive quadrupedal locomotion, offering new insights into dynamic gait generation and control.

#### II. RELATED WORKS

#### A. Symmetry in Locomotion and Control

Symmetry has long served as a foundation for locomotion analysis and controller design. Hildebrand [6], [7] classified quadrupedal gaits by inter-limb phase relations, showing that symmetrical and asymmetrical gaits form continuous families that enable smooth transitions. Raibert [5] emphasized time-reversal symmetry in running robots, where forward and backward motions are mirror images. Razavi et al. [4] applied odd-even symmetry to generate efficient periodic gaits, and Ordonez et al. [8] demonstrated that exploiting discrete symmetries improves neural-network sample efficiency. Ding et al. [9] showed that symmetry breaking expands quadrupedal gait diversity. More recently, Su et al. [10] incorporated equivariance constraints into RL architectures, improving gait quality and sim-to-real robustness. Symmetry has also been exploited in perception tasks, where Butterfield et al. [11] designed a morphology-informed graph neural network for contact estimation, highlighting its broader role in robot learning.

### B. Reinforcement Learning for Locomotion

RL-based locomotion methods can be categorized into reference-based and reference-free approaches. Reference-based methods rely on predefined templates. On Cassie, a SLIP-based gait library supported walking up to 1.2 m/s [12], and RL feedback enabled robust disturbance recovery [13]. In quadrupeds, CPG frameworks encode phase shifts and duty factors to parameterize cyclic motions [14], [15], but they constrain robots to handcrafted templates. Reference-free methods instead rely on enforcing specific properties of gaits. Early works were limited to single gaits, such as walking towards a certain direction [16] or trotting based

on specified commands [17]. Later, Siekmann *et al.* [18] enforced the periodicity of foot ground reaction forces and velocities to learn all bipedal gaits, and Margolis and Agrawal [19] expanded the command space to diversify quadrupedal gaits. These designs achieve variety but require careful reward tuning to maintain training stability.

Although symmetry has proven effective for both locomotion analysis and learning, most RL-based gait generation still depends on explicit models or command-space augmentation. Our framework differs by embedding temporal, time-reversal, and morphological symmetries directly into the reward design. This inductive bias enforces structural invariances of gait families, reduces the number of gait parameters, and enables dynamic quadrupedal gaits that generalize across speeds and transitions without predefined foot sequences.

#### III. SYMMETRY-GUIDED REINFORCEMENT LEARNING

We present a unified framework to explore quadrupedal locomotion across symmetry classes using a set of principal gait parameters. The method integrates symmetry-aware gait specification, a Markov decision process formulation, structured state—action design, and a reward combining command tracking, smoothness, and symmetry terms. As illustrated in Fig. 2, this yields robust policies that generalize across gaits and commanded speeds.

# <span id="page-1-1"></span>A. Gait Specification

A systematic parameterization of quadrupedal gaits provides the foundation for generating trajectories and analyzing symmetry properties in our framework. We model a gait as the periodic orbit of a hybrid dynamical system [9], where each cycle alternates between stance and swing phases over a stride *period* T. The normalized *phase*  $\phi \in [0,1)$  denotes the position within the cycle, and the phase of each leg  $i \in \{LH, LF, RF, RH\}$  (left hind, left front, right front, right hind, respectively) is shifted by a leg-specific *phase offset*  $\theta_i \in [0,1)$ . The fraction of the stride in stance is described

<span id="page-2-1"></span>

| Gait              | $\theta_{\rm LH}$ | $\theta_{\rm LF}$ | $\theta_{\rm RF}$ | $\theta_{\mathrm{RH}}$ |
|-------------------|-------------------|-------------------|-------------------|------------------------|
| Trotting          | 0.00              | 0.50              | 0.00              | 0.50                   |
| Bounding          | 0.00              | 0.50              | 0.50              | 0.00                   |
| Half-bounding (L) | 0.00              | 0.63              | 0.37              | 0.00                   |
| Half-bounding (R) | 0.00              | 0.37              | 0.63              | 0.00                   |
| Rotary gallop     | 0.13              | 0.37              | 0.63              | 0.87                   |
| Transverse gallop | 0.13              | 0.63              | 0.37              | 0.87                   |

TABLE I: Phase offsets  $\theta_i$  for representative quadrupedal gaits. LH, LF, RF, RH denote left hind, left front, right front, and right hind legs.

by the *duty factor*  $\beta \in (0,1)$ . The stride period T and duty factor  $\beta$  are intrinsically coupled to forward velocity. Fixing them limits attainable speeds, while arbitrary randomization hinders convergence. Prior studies [9], [20], [21] showed that these temporal parameters align with oscillatory modes from body–limb energy exchanges. To capture this dependency, we combine insights from the passive dynamics of a quadrupedal SLIP model [9], [22] with empirical calibration from RL trials, yielding

$$T^* = a_T (1 + b_T \delta |v_x^*|^{\text{cmd}}) e^{-c_T |v_x^*|^{\text{cmd}}},$$
  

$$\beta = a_\beta (1 + b_\beta \delta |v_x^*|^{\text{cmd}}) e^{-c_\beta |v_x^*|^{\text{cmd}}},$$
(1)

where  $\delta \sim U(-1,1)$  introduces small uniform perturbations,  $v_x^*$  cmd is the command velocity normalized by  $\sqrt{gl}$ , and g and l are gravity and leg length, respectively. The fitted constants  $(a_T,b_T,c_T,a_\beta,b_\beta,c_\beta)$  are reported in Section III-E. For tractability, we assume  $\beta$  is identical across legs. We analyze four gait families with symmetry-based variants (Fig. 3), summarized in Table I. These symmetry distinctions determine the structure of feasible trajectories and serve as constraints in subsequent optimization. Under the above definitions and assumptions, we specify a gait using its leg phase offsets and forward command velocity

$$\Gamma := \left[\theta_{LH}, \theta_{LF}, \theta_{RF}, \theta_{RH}, v_x^{\text{cmd}}\right]. \tag{2}$$

<span id="page-2-0"></span>![](_page_2_Figure_6.jpeg)

Fig. 3: Footfall patterns of four quadrupedal gaits. Colored bars denote stance and blanks denote swing. LH, LF, RF, RH indicate left hind, left front, right front, and right hind legs. These examples span the symmetry classes analyzed.

## B. Problem Formulation

We formulate our problem within the context of a discretetime Markov Decision Process (MDP) defined by the tuple

<span id="page-2-2"></span>

| Term                                                                               | Expression                                                                                                                                                                                                                          |
|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| x velocity tracking y velocity tracking yaw velocity tracking base height tracking | $\begin{array}{l} -0.3(1-\exp\{-2 v_x-v_x^{\mathrm{cmd}} \}) \\ -0.3(1-\exp\{-10 v_y-v_y^{\mathrm{cmd}} \}) \\ -0.3(1-\exp\{-5 \omega_{\mathrm{yaw}}-\omega_{\mathrm{yaw}}^{\mathrm{cmd}} \}) \\ -0.3(1-\exp\{-5d_h\}) \end{array}$ |
| torque differences<br>hip action smoothness<br>swing-phase clearance               | $-0.1(1 - \exp\{-0.1 \  \boldsymbol{\tau}_t - \boldsymbol{\tau}_{t-1} \ _1\}) \\ -0.15(1 - \exp\{-0.5 \sum_{t}  \boldsymbol{a}_t^{hip} \}) \\ -0.15I_{swing}(\phi_i)(1 - \exp\{-20c_{foot}\})$                                      |

TABLE II: Reward terms for command tracking  $R_{\rm cmd}$  (top) and smoothness  $R_{\rm smooth}$  (bottom).

 $(\mathcal{S},\mathcal{A},R,P,\gamma)$ . In an MDP,  $\mathcal{S}$  and  $\mathcal{A}$  represent the continuous state and action space respectively, and  $\gamma \in (0,1)$  denotes the discount factor. At any time step t, the agent selects an action  $a_t \in \mathcal{A}$  to move from the current state  $s_t \in \mathcal{S}$  to the next state  $s_{t+1} \in \mathcal{S}$ , following the transition probability  $P(s_{t+1}|s_t,a_t)$  and returning a scalar reward  $R(s_t,a_t)$ . The goal is to discover the optimal control policy  $\pi(a_t|s_t)$  that maximizes the expectation of the discounted return  $\mathcal{J}(\pi)$  over any trajectories induced by  $\pi$ :

$$\mathcal{J}(\pi) = \mathbb{E}_{\boldsymbol{a}_t \sim \pi(\cdot|\boldsymbol{s}_t)} \left[ \sum_{t=0}^{\infty} \gamma^t R(\boldsymbol{s}_t, \boldsymbol{a}_t) \right]. \tag{3}$$

#### <span id="page-2-3"></span>C. State and Action Space

We implement our framework on a 12 degree-of-freedom Unitree quadrupedal robot Go2, where each leg has a hip, thigh and calf joint. The action space  $\mathcal{A} \in \mathbb{R}^{12}$  is the target position of 12 joints, and the position control is achieved via a PD controller with  $k_p=30$  and  $k_d=0.65$ . The state space  $\mathcal{S} \in \mathbb{R}^{52}$  encompasses the observed robot states, gait parameters and user commands. To be specific, the state vector  $\mathbf{s}_t \in \mathcal{S}$  at time step t includes proprioception (orientation  $\mathbf{g}^{\text{ori}} \in \mathbb{R}^3$ , joint positions  $\mathbf{q} \in \mathbb{R}^{12}$  and velocities  $\dot{\mathbf{q}} \in \mathbb{R}^{12}$ ), last action  $\mathbf{a}_{t-1} \in \mathbb{R}^{12}$ , velocity commands  $(v_x^{\text{cmd}}, v_y^{\text{cmd}}, \omega_{\text{yaw}}^{\text{cmd}})$ , the clock input of four legs  $\mathbf{p} = \{\sin(2\pi\phi_i)), i \in \{\text{LH, LF, RF, RH}\}$  using the current leg phase  $\phi_i$ , the phase offset of legs for indicating the desired gait  $\boldsymbol{\theta} = [\theta_{\text{LF}}, \theta_{\text{RF}}, \theta_{\text{LH}}, \theta_{\text{RH}}]$ , and the ratio of stance and swing phase  $\mathbf{r} = [\beta, 1 - \beta]$ .

## D. Reward Function Design

As shown in Fig. 2, the reward function combines multiple objectives: tracking user commands, encouraging smooth and energetically efficient motions, and enforcing symmetries. Each component is scaled to similar magnitudes to stabilize training, and the total reward is clipped to be non-negative:

$$R = \max\{0, 1 + R_{cmd} + R_{smooth} + R_{sym}\}.$$
 (4)

Table II introduces all the terms included in the command tracking reward  $R_{\rm cmd}$  and the smoothness reward  $R_{\rm smooth}$ . The command tracking reward  $R_{\rm cmd}$  penalizes deviations from commanded base velocities in x, y, and yaw, and encourages the torso height  $h_{\rm base}$  to remain within the range  $[h_{\rm base}^{\rm min}, h_{\rm base}^{\rm max}]$  by penalizing boundary violations:

$$d_h = \max\{0, h_{\text{base}}^{\min} - h_{\text{base}}\} + \max\{0, h_{\text{base}} - h_{\text{base}}^{\max}\}.$$
 (5)

The smoothness reward  $R_{\rm smooth}$  penalizes rapid torque changes, excessive hip motions, and insufficient foot clearance during swing. Here  $a_t^{\rm hip}$  is the softmax-weighted magnitude of hip joint actions, the binary indicator  $I_{\rm swing}(\phi_i)$  equals 1 if leg i is in swing at phase  $\phi_i$  and 0 otherwise, and  $c_{\rm foot}$  quantifies clearance shortfall:

$$c_{\text{foot}} = \sum_{i=1}^{4} w(s_i) \max(0, h_{\text{cl}}^{\text{min}} - z_i),$$

$$w(s_i) = \frac{1}{2} (1 + \sin(\pi s_i))$$
(6)

where  $z_i$  is the vertical position of foot i,  $h_{\rm cl}^{\rm min}$  the desired minimal clearance, and  $s_i = {\rm clip}\Big(\frac{\phi_i}{1-\beta},\,0,\,1\Big)$ . Besides, we incorporate three types of symmetry observed

Besides, we incorporate three types of symmetry observed in animal and robotic locomotion: temporal, morphological, and time-reversal. Time-reversal symmetry is not expressed as an explicit reward but enforced via phase mapping, and the final symmetry reward is defined as  $R_{\rm sym} = R_{\rm tem} + R_{\rm mor}$ .

1) Temporal symmetry: To ensure each leg exhibits a single stance per stride and avoid Zeno-like switching [23], we gate penalties with phase-dependent stance/swing indicators. For leg i, stance is defined over  $[1-\beta,1)$  and swing over  $[0,1-\beta)$ . To smooth the phase transition, we introduce the Von Mises distribution to the binary indicators  $I_{\rm swing}$  and  $I_{\rm stance}$  and calculate their expectation [18]. The reward penalizes high velocity when feet are desired to be stance, and nonzero ground reaction forces (GRFs) when feet are desired to be swing

$$R_{\text{tem}} = -0.15 \sum_{i} \left( \mathbb{E}[I_{\text{swing}}(\phi_i)] (1 - \exp\{-0.001 \| \boldsymbol{f}_i \| \}) + \mathbb{E}[I_{\text{stance}}(\phi_i)] (1 - \exp\{-2 \| \boldsymbol{v}_i \| \}) \right),$$

$$(7)$$

with  $f_i$  the GRF and  $v_i$  the foot velocity of leg i.

2) Morphological symmetry: When two legs share the same phase offset  $(\theta_i = \theta_j)$ , they should produce similar joint trajectories. We use a tolerance  $\epsilon_{\sigma} = 0.01$  to detect such symmetry and penalize deviations:

$$R_{\text{mor}} = -0.15 \left( 1 - \exp\{-5d(G_{\sigma})\} \right),$$

$$d(G_{\sigma}) = \sum_{\substack{\sigma(i,j) \in G_{\sigma} \\ k \in \{\text{hip,thigh,knee}\}}} f(\sigma(i,j)) |q_{i,k} - q_{j,k}|, \quad (8)$$

where  $q_{(\cdot,\cdot)}$  are joint positions and  $f(\sigma(i,j))=1$  if  $|\theta_i-\theta_j| \le \epsilon_\sigma$  and 0 otherwise. This discourages uneven leg usage and prevents limping behaviors.

3) Time-reversal symmetry: Quadrupedal locomotion exhibits approximate invariance under reversing time and direction [5], [9]. To enforce this property for leg i, we remap its phase in time-reversal style when the commanded velocity is negative, and the modified leg phase  $\phi_i$  is

<span id="page-3-2"></span>
$$\phi_i = \begin{cases} (\phi + \theta_i) \mod 1 & \text{if} \quad v_x^{\text{emd}} \ge 0 \\ -(\phi + \theta_i) \mod (-1) & \text{if} \quad v_x^{\text{emd}} < 0 \end{cases}$$
(9)

where mod is the modulo function. This guarantees that backward motion mirrors forward motion, preventing policies from favoring a subset of legs when switching direction.

<span id="page-3-1"></span>

| Type         | Parameter                                                 | Range                                                            |
|--------------|-----------------------------------------------------------|------------------------------------------------------------------|
| Dynamics     | Body Mass<br>Body Friction<br>Torso Velocity Perturbation | $[-1.5, 1.5]$ kg $[0.3, 2.0] \times$ default $[-0.25, 0.25]$ m/s |
| State Noises | Orientation Joint Position Joint Velocity                 | [-0.05, 0.05]<br>[-0.01, 0.01] rad<br>[-1.5, 1.5] rad/s          |

TABLE III: Randomized training parameters. Each of the 1024 Isaac Gym environments is assigned randomized dynamics at initialization, and uniform noise is injected into the state vector at every timestep.

#### <span id="page-3-0"></span>E. Reinforcement Learning and Hardware Setup

We train policies using Proximal Policy Optimization (PPO) [24]. The actor is a multi-layer perceptron (MLP) with hidden sizes [512, 256, 128] and ELU activations; the critic shares the architecture but outputs a scalar state value. Control runs at 50 Hz. The hyperparameters of PPO are: learning rate with the initial value 0.001 and adaptively updated by Adam [25], discount  $\gamma = 0.99$ , GAE factor  $\lambda =$ 0.95 [26], clipping threshold  $\epsilon = 0.2$ , and entropy weight 0.01. Training is conducted in Isaac Gym [27] with 2048 agents in parallel. The policy is updated every 24 simulation steps and training stops after  $2.46 \times 10^9$  samples. Domain randomization is applied at initialization and observation noise at each step (Table III). Parameters for the gait-period and duty-factor formulation in Eq. 1 are fixed to  $a_T = 2.55$ ,  $b_T = 0.20, c_T = 0.975, a_\beta = 0.5588, b_\beta = 0.20, and$  $c_{\beta} = 0.681$ , with the torso height randomized uniformly in [0.35, 0.45] m.

Episodes last  $30\,\mathrm{s}$  with both velocity and gait resampling. At the start, forward velocity  $v_x^\mathrm{cmd}$  is drawn from  $[-2,2]\,\mathrm{m/s}$ , while  $v_y^\mathrm{cmd} = \omega_{\mathrm{yaw}}^\mathrm{cmd} = 0$ . At  $t = 10\,\mathrm{s}$ ,  $v_x^\mathrm{cmd}$  is resampled to encourage adaptation. For gait selection, an initial gait is chosen from the library with uniform noise  $\pm 0.02$  on each phase parameter  $\theta$ ; at  $t = 20\,\mathrm{s}$ , a new gait is sampled to enforce transitions. Policies are deployed on the Unitree Go2 via wired connection. Lightweight Communications and Marshalling [28] links the robot to a laptop with an RTX 4090 GPU, which reads robot and joystick data, executes the trained policy, and sends joint-level commands back. No hand-tuned gait schedules are used at deployment; gait commands are provided only through joystick inputs.

## F. Ablation Study Setup

We evaluate the contribution of each symmetry component by selectively disabling them during training and testing. To study morphological symmetry, we remove the joint-similarity penalty so that gait evaluation depends only on stance and swing agreement, without enforcing coordination between symmetric limbs. To study time-reversal symmetry, we disable the mapping in Eq. 9 by omitting the phase flip  $\phi$  when  $v_x^{\rm cmd} < 0$ , forcing forward and backward motion to share the same phase schedule.

#### G. Performance Metrics

Policies are evaluated using the three following metrics:

1) Command velocity tracking: Per-episode root-meansquare normalized error (RMSNE) between commanded and realized base velocities:

$$\text{RMSNE}_{\text{cmd}} = \left[ \frac{1}{T} \sum_{t=1}^{T} \sum_{d \in \{x, y, \omega_z\}} \left( \frac{v_d^{\text{cmd}} - v_d(t)}{|v_d^{\text{cmd}}| + \varepsilon} \right)^2 \right]^{1/2},$$
(10)

where  $\varepsilon$  is a small constant to avoid division by zero.

2) Gait consistency: Mismatch between desired and realized stance/swing states, with an additional term penalizing asymmetry across symmetric leg pairs. The stance consistency term is

$$GC_{\text{stance}} = \frac{1}{T} \sum_{t=1}^{T} \left[ 1 - \frac{1}{4} \sum_{i=1}^{4} \mathbf{1}(s_i^{\text{real}}(t) = s_i^{\text{des}}(t)) \right], \quad (11)$$

and the morphological consistency term is

$$GC_{morph} = \frac{1}{T} \sum_{t=1}^{T} \sum_{(a,b) \in \mathcal{P}} ||q_a(t) - q_b(t)||, \qquad (12)$$

where  $\mathcal{P}$  denotes selected symmetric leg pairs and  $q_a$  the joint positions. The combined measure is

$$GC = \frac{1}{T} \sum_{t=1}^{T} (GC_{\text{stance}}(t) + GC_{\text{morph}}(t)).$$
 (13)

3) Cost of transport (CoT): Normalized positive mechanical work relative to weight and distance traveled:

$$CoT = \frac{\int_0^T \sum_j |\tau_j(t) \, \dot{q}_j(t)|_+ dt}{mg \, \Delta x}, \tag{14}$$

where  $\tau_j$  and  $\dot{q}_j$  denote the torque and joint velocity of actuator j, mg is the robot's weight, and  $\Delta x$  is the net forward displacement over the episode.

## IV. RESULTS

We assess the proposed symmetry-guided RL framework in simulation and hardware. The evaluation covers: (A) velocity tracking across commanded speeds, (B) gait tracking and transitions among trotting, bounding, half-bounding, and galloping, (C) ablation of morphological symmetry, and (D) hardware validation on the Unitree Go2. These results demonstrate accurate tracking, versatile gait generation, and reliable sim-to-real transfer.

# A. Velocity Tracking

We first evaluated velocity tracking in simulation across four gaits. The commanded forward velocity  $v_x^{\rm cmd}$  was initialized at -2 [m/s] and changed sequentially to -1, 0.2, and 2 [m/s] at t=5, 10, and 15 [s], respectively, while the lateral command was fixed at  $v_y^{\rm cmd}=0$ . As shown in Fig. 4, the actual velocity converged to each commanded value within approximately 1 [s], demonstrating stable and responsive tracking. Performance was consistent across all gaits, indicating that the learned policy achieves equally accurate velocity regulation independent of footfall sequence.

<span id="page-4-0"></span>![](_page_4_Figure_16.jpeg)

Fig. 4: Velocity tracking across four gaits under varying commanded forward speeds. (A) Sequential transitions with  $v_x^{\rm cmd} \in \{-2, -1, 0.2, 2\}$  [m/s]. (B) Tracking of  $v_y^{\rm cmd} = 0$  during the same test.

#### B. Gait Tracking

We evaluated gait tracking under a constant forward velocity of  $v_x^{\rm cmd} = 0.5$  [m/s]. The robot sequentially executed trotting, bounding, half-bounding, and galloping, each lasting 2 [s]. Fig. 5(A) shows representative frames of the four gaits, including transition phases. Fig. 5(B) compares desired footfall sequences (hollow) with simulated contact sequences (solid). All transitions were achieved within a single step (approximately 0.5 [s]). The mean absolute error between desired and realized contacts was  $GC_{stance} = 0.07$ , indicating accurate tracking. The learned policy generalized across arbitrary gait transitions and commanded velocities, while maintaining stable body posture without collapse or lateral drift. Supplementary demonstrations, including left- versus right-leading leg switches in half-bounding and galloping, and abrupt transitions from galloping to trotting, are provided in the online repository<sup>1</sup>.

# C. Ablation Study Results

We assessed the role of morphological and time-reversal symmetries by comparing the full symmetry-enforced policy (ours) against variants trained with individual symmetry terms removed. Performance was evaluated using three metrics averaged across all gaits introduced in Sec. III-A, as summarized in Fig. 6.

Command velocity tracking: The full policy achieved the lowest RMSNE across positive speeds, demonstrating the most accurate forward tracking. Removing morphological symmetry caused noticeable degradation, while removing time-reversal symmetry mainly increased errors at negative speeds, confirming its importance for backward motion.

Gait consistency: The full policy achieved an average GC error of about 0.2, whereas the no-morphological variant rose to about 0.4. Removing time-reversal symmetry had little impact on forward gaits but reduced consistency during reversals. These results highlight that morphological symmetry primarily governs inter-leg coordination, while time-reversal symmetry ensures mirrored behaviors when switching direction.

<span id="page-4-1"></span><sup>&</sup>lt;sup>1</sup>Supplementary material and videos: https://anonymous.4open.science/r/go2\_symm\_r1-F623/

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Fig. 5: Gait tracking at  $v_x^{\rm cmd} = 0.5$  [m/s]. (A) Representative frames of trotting, bounding, half-bounding, and galloping, including transition phases. (B) Desired footfall sequences (hollow bars) compared with realized touchdown sequences in simulation (solid bars).

Cost of transport: The full policy achieved comparable CoT to both the no-morphology and no-time-reversal variants when moving at negative velocities, but showed consistently lower CoT for positive velocities. For instance, at  $v_{\rm cmd}=0.2$  m/s, the full policy reached CoT = 2.86, compared to 3.47 for the no-time-reversal and 4.15 for the no-morphology variants. At  $v_{\rm cmd}=0.5$  m/s, the three policies yielded CoT values of 1.10, 1.25, and 1.39, respectively. Overall, the no-time-reversal variant incurred roughly 10–20% higher energy cost than the full policy, while the no-morphology variant was 30–50% more costly. These results indicate that morphological symmetry has the strongest influence on energetic efficiency, whereas time-reversal symmetry primarily enhances directional robustness.

Overall, these results show that both morphological and time-reversal symmetries play distinct but complementary roles. Morphological symmetry drives coordination and efficiency, while time-reversal symmetry guarantees consistent bidirectional locomotion. Their combination yields the most accurate, coordinated, and efficient gaits.

<span id="page-5-1"></span>![](_page_5_Figure_4.jpeg)

Fig. 6: Effect of ablating morphological/time-reversal symmetry: (A) command velocity tracking, (B) gait consistency, and (C) CoT. The symmetry-enforced policy consistently outperforms the no-symmetry variant in accuracy, coordination, and efficiency.

## D. Hardware Performance

We validated the learned policy on the Unitree Go2. A constant forward command of  $v_x^{\text{cmd}} = 0.8 \text{ [m/s]}$  was applied, while gaits were switched sequentially from trotting to bounding, half-bounding, and galloping, each lasting 5 [s]. Figure 7(A) compares desired and realized contact sequences. In all cases the robot reproduced the expected footfall patterns. For trotting and galloping, contact timing closely matched the reference. In bounding and halfbounding, the paired limbs maintained correct synchronization and phasing, but contacts deviated by about 0.1 [s] from the desired sequence. Despite these shifts, all gaits executed stably without collapse or lateral drift. Keyframes of the four gaits are shown in Fig. 7(B) consistent with the desired sequences. The robot converged to the commanded velocity within 1 [s], demonstrating reliable tracking. Robustness was further tested by executing bounding on uneven terrain: as shown in Fig. 7(D), the robot transitioned smoothly from concrete to grass without failure, confirming that the policy generalizes to outdoor conditions and maintains stability under perturbations.

## V. CONCLUSIONS

This work introduced a symmetry-guided reinforcement learning framework that unifies quadrupedal gait generation without relying on predefined trajectories or gait-specific controllers. By embedding temporal, morphological, and time-reversal symmetries directly into the reward design, a single policy reproduced trotting, bounding, half-bounding, and galloping on the Unitree Go2 robot, achieving accurate velocity tracking, coordinated footfall patterns, and improved energetic efficiency. The use of period and duty factor sampling curves further enabled scalable gait modulation, supporting more dynamic locomotion across speeds. Simulation and hardware results confirmed that symmetries enhance the adaptability and robustness across diverse gaits.

# REFERENCES

[1] D. Kim, J. Di Carlo, B. Katz, G. Bledt, and S. Kim, "Highly dynamic quadruped locomotion via whole-body impulse control and model predictive control," 2019. [Online]. Available: https: //arxiv.org/abs/1909.06586

<span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

Fig. 7: Hardware validation on the Unitree Go2: (A) Comparison of desired (hollow) and true (solid) contact sequences across trotting, bounding, half-bounding, and galloping. (B) Velocity tracking at  $v_x^{\rm cnd} = 0.8$  [m/s], where the true velocities are predicted by a state estimator. (C) Keyframes of robot applying the four gaits. (D) Robustness test on uneven terrain, where bounding gaits remained stable during transitions from concrete to grass.

- [2] Z. Fu, X. Cheng, and D. Pathak, "Deep whole-body control: learning a unified policy for manipulation and locomotion," in *Conference on Robot Learning*. PMLR, 2023, pp. 138–149.
- [3] M. Hildebrand, "Analysis of asymmetrical gaits," *Journal of Mammalogy*, vol. 58, no. 2, pp. 131–156, May 1977. [Online]. Available: https://doi.org/10.2307/1379571
- [4] H. Razavi, A. M. Bloch, C. Chevallereau, and J. W. Grizzle, "Symmetry in legged locomotion: a new method for designing stable periodic gaits," *Autonomous Robots*, vol. 41, pp. 1119–1142, 2017.
- [5] M. H. Raibert, "Running with symmetry," *The International Journal of Robotics Research*, vol. 5, no. 4, pp. 3–19, Dec. 1986. [Online]. Available: https://doi.org/10.1177/027836498600500401
- [6] M. Hildebrand, "Symmetrical gaits of horses," *Science*, vol. 150, no. 3697, pp. 701–708, Nov. 1965. [Online]. Available: https://doi.org/10.1126/science.150.3697.701
- [7] —, "The quadrupedal gaits of vertebrates," *BioScience*, vol. 39, no. 11, pp. 766–775, Dec. 1989. [Online]. Available: https://doi.org/10.2307/1311182
- [8] D. Ordonez-Apraez, M. Martin, A. Agudo, and F. Moreno-Noguer, "On discrete symmetries of robotics systems: A group-theoretic and data-driven analysis," arXiv preprint arXiv:2302.10433, 2023.
- [9] J. Ding and Z. Gan, "Breaking symmetries leads to diverse quadrupedal gaits," *IEEE Robotics and Automation Letters*, vol. 9, no. 5, p. 4782–4789, May 2024. [Online]. Available: http://dx.doi.org/10.1109/LRA.2024.3384908
- [10] Z. Su, X. Huang, D. Ordoñez-Apraez, Y. Li, Z. Li, Q. Liao, G. Turrisi, M. Pontil, C. Semini, Y. Wu, and K. Sreenath, "Leveraging symmetry in rl-based legged locomotion control," in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024, pp. 6899–6906.
- [11] D. Butterfield, S. S. Garimella, N.-J. Cheng, and L. Gan, "Mi-hgnn: Morphology-informed heterogeneous graph neural network for legged robot contact perception," in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 10110–10115.
- [12] K. Green, Y. Godse, J. Dao, R. L. Hatton, A. Fern, and J. Hurst, "Learning spring mass locomotion: Guiding policies with a reduced-order model," *IEEE Robotics and Automation Letters*, vol. 6, no. 2, p. 3926–3932, Apr. 2021. [Online]. Available: http://dx.doi.org/10.1109/LRA.2021.3066833
- [13] Z. Xie, G. Berseth, P. Clary, J. Hurst, and M. van de Panne, "Feedback control for cassie with deep reinforcement learning," in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, Oct. 2018. [Online]. Available: http://dx.doi.org/10.1109/IROS.2018.8593722
- [14] G. Schöner, W. Jiang, and J. Kelso, "A synergetic theory of quadrupedal gaits and gait transitions," *Journal of Theoretical Biology*, vol. 142, no. 3, pp. 359–391, Feb. 1990. [Online]. Available: https://doi.org/10.1016/s0022-5193(05)80558-2
- [15] Y. Shao, Y. Jin, X. Liu, W. He, H. Wang, and W. Yang, "Learning free gait transition for quadruped robots via phase-guided controller."

- IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 1230–1237, 2021.
- [16] T. Haarnoja, S. Ha, A. Zhou, J. Tan, G. Tucker, and S. Levine, "Learning to walk via deep reinforcement learning," in *Robotics: Science and Systems XV*, ser. RSS2019. Robotics: Science and Systems Foundation, June 2019. [Online]. Available: http://dx.doi.org/10.15607/RSS.2019.XV.011
- [17] N. Kohl and P. Stone, "Policy gradient reinforcement learning for fast quadrupedal locomotion," in *IEEE International Conference* on Robotics and Automation, 2004. Proceedings. ICRA '04. 2004. IEEE, 2004. [Online]. Available: http://dx.doi.org/10.1109/ROBOT. 2004.1307456
- [18] J. Siekmann, Y. Godse, A. Fern, and J. Hurst, "Sim-to-real learning of all common bipedal gaits via periodic reward composition," in 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, May 2021. [Online]. Available: http://dx.doi.org/10. 1109/ICRA48506.2021.9561814
- [19] G. B. Margolis and P. Agrawal, "Walk these ways: Tuning robot control for generalization with multiplicity of behavior," in *Conference* on Robot Learning. PMLR, 2023, pp. 22–31.
- [20] Z. Gan, Z. Jiao, and C. D. Remy, "On the dynamic similarity between bipeds and quadrupeds: A case study on bounding," *IEEE Robotics and Automation Letters*, vol. 3, no. 4, pp. 3614–3621, 2018.
  [21] Y. G. Alqaham, J. Cheng, and Z. Gan, "Energetic analysis on the
- [21] Y. G. Alqaham, J. Cheng, and Z. Gan, "Energetic analysis on the optimal bounding gaits of quadrupedal robots," 2023.
- [22] Z. Gan, T. Wiestner, M. A. Weishaupt, N. M. Waldern, and C. David Remy, "Passive dynamics explain quadrupedal walking, trotting, and tölting," *Journal of computational and nonlinear dynamics*, vol. 11, no. 2, p. 021008, 2016.
- [23] A. D. Ames, A. Abate, and S. Sastry, "Sufficient conditions for the existence of zeno behavior," in *Proceedings of the 44th IEEE Conference on Decision and Control*. IEEE, 2005, pp. 696–701.
- [24] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv preprint arXiv:1707.06347, 2017.
- [25] K. D. B. J. Adam et al., "A method for stochastic optimization," arXiv preprint arXiv:1412.6980, vol. 1412, no. 6, 2014.
- [26] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, "High-dimensional continuous control using generalized advantage estimation," arXiv preprint arXiv:1506.02438, 2015.
- [27] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State, "Isaac gym: High performance gpu-based physics simulation for robot learning," 2021.
- [28] A. S. Huang, E. Olson, and D. C. Moore, "Lcm: Lightweight communications and marshalling," in 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2010, pp. 4057– 4062.

---

## Notes

- **Title:** Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy Enables Free Gait Transitions at Varying Speeds
- **URL:** https://arxiv.org/pdf/2510.10455v1
