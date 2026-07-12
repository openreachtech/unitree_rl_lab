# PGTT: Phase-Guided Terrain Traversal for Perceptive Legged Locomotion

Alexandros Ntagkas<sup>1,2</sup>, Chairi Kiourt<sup>2,3</sup>, and Konstantinos Chatzilygeroudis<sup>1,2,4</sup>

Abstract—State-of-the-art perceptive Reinforcement Learning controllers for legged robots either (i) impose oscillator or IK-based gait priors that constrain the action space, add bias to the policy optimization and reduce adaptability across robot morphologies, or (ii) operate "blind", which struggle to anticipate hind-leg terrain, and are brittle to noise. In this paper, we propose Phase-Guided Terrain Traversal (PGTT), a perception-aware deep-RL approach that overcomes these limitations by enforcing gait structure purely through reward shaping, thereby reducing inductive bias in policy learning compared to oscillator/IK-conditioned action priors. PGTT encodes per-leg phase as a cubic Hermite spline that adapts swing height to local heightmap statistics and adds a swing-phase contact penalty, while the policy acts directly in joint space supporting morphology-agnostic deployment. Trained in MuJoCo (MJX) on procedurally generated stair-like terrains with curriculum and domain randomization, PGTT achieves the highest success under push disturbances (median +7.5% vs. the next best method) and on discrete obstacles (+9%), with comparable velocity tracking, and converging to an effective policy roughly 2× faster than strong end-to-end baselines. We validate PGTT on a Unitree Go2 using a real-time LiDAR elevation-to-heightmap pipeline, and we report preliminary results on ANYmal-C obtained with the same hyperparameters. These findings indicate that terrain-adaptive, phase-guided reward shaping is a simple and general mechanism for robust perceptive locomotion across platforms.

#### I. Introduction

Legged robots promise unmatched mobility in cluttered, uneven, and human-made environments, but robust gait control on such terrain remains challenging [1], [2]. Reinforcement learning (RL) has shown that agile locomotion behaviors can be learned from data [3], yet many studies assume *idealized sensing* (privileged terrain information) or operate "blind," which hinders anticipation of obstacles and reduces reliability on hardware [4], [5]. As a result, perception is essential, but the representation and how it interfaces with control are pivotal for generality and robustness.

\*This work was supported by the Hellenic Foundation for Research and Innovation (H.F.R.I.) under the "3rd Call for H.F.R.I. Research Projects to support Post-Doctoral Researchers" (Project Acronym: NOSALRO, Project Number: 7541). This work has also been partially supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the NextGenerationEU Program.

<sup>1</sup>Laboratory of Automation and Robotics (LAR) in the Department of Electrical & Computer Engineering, University of Patras, GR-26504 Patras, Greece, a\_ntagkas@ac.upatras.gr, costashatz@upatras.gr

<sup>2</sup>Archimedes/Athena RC, Greece

 $^3A$ thena - Research and Innovation Center in Information, Communication and Knowledge Technologies, Xanthi, Greece, chairiq@athenarc.gr

<sup>4</sup>Computational Intelligence Laboratory (CILab), Department of Mathematics, University of Patras, GR-26110 Patras, Greece

![](_page_0_Picture_11.jpeg)

Fig. 1: Real-world example of the Unitree Go2 robot climbing stair terrain.

Recent efforts incorporate visual or range sensing into the loop. Egocentric depth cameras enable end-to-end training and have demonstrated stair and gap traversal [6], but limited field of view, sensor noise, and the need for temporal memory remain practical obstacles (especially for hind-leg terrain). Methods that rely on globally consistent elevation maps or multi-sensor rigs can extend foresight but require careful calibration and accurate global pose estimation, which can be brittle [7]. In parallel, many RL controllers encode *gait priors* by prescribing oscillatory foot/joint targets as functions of a per-leg phase and tracking them with inverse kinematics (IK) and PD controllers; while effective, this constrains the action space and couples policies to specific morphologies, introducing bias that may reduce adaptability [4], [8].

We propose **Phase-Guided Terrain Traversal (PGTT)**, a perception-aware deep-RL approach that retains the benefits of rhythmic structure while avoiding IK and action-space constraints. PGTT uses a robot-centric *heightmap* (derived online from LiDAR elevation mapping) as a compact terrain representation and encodes per-leg phase with a *cubic Hermite spline* whose swing apex adapts to local height statistics. Crucially, the phase prior is enforced *only through reward shaping*, while the policy acts directly in joint space. This design keeps the action space unconstrained and *reduces inductive bias* in policy learning compared to oscillator/IK-conditioned targets, easing deployment across different morphologies.

Across extensive simulation tests, PGTT exhibits faster learning convergence than an end-to-end, non-prior RL baseline and delivers consistently higher survival rates under pushes and on discrete obstacles than state-of-the-art perceptive locomotion methods. The gains in survival come without sacrificing tracking quality, as velocity-tracking performance remains comparable to the strongest baselines. We then deploy the learned policy on a Unitree Go2 with a real-time elevation-to-heightmap pipeline and demonstrate robust stair and obstacle traversal, and we present preliminary results on **ANYmal** C obtained *without* 

*changing the hyper-parameters*. These findings support the central premise of phase-guided reward shaping: it provides useful structure for learning while keeping the action space unconstrained and thus more adaptable across platforms.

The main contributions of this manuscript are:

- A terrain-adaptive, phase-guided *reward* that encodes a Hermite-spline swing trajectory driven by local heightmap statistics and penalizes swing-phase contacts, without constraining the action space or using IK, thereby reducing inductive bias and improving morphology-agnostic deployment.
- A compact perception-to-policy design that feeds a robot-centric heightmap directly to the policy, avoiding global pose assumptions while capturing nearby terrain geometry relevant for stepping [5].
- An extensive evaluation on stair-like and discrete-obstacle terrains showing higher success rates, Sim2Real transfer on a Go2 quadruped, and preliminary cross-platform results on ANYmal C with zero hyper-parameter changes.
- An accessible training stack using MuJoCo/MJX that provides accurate dynamic simulation and high throughput on a single consumer GPU, offering a lightweight alternative to Isaac Gym-based pipelines [9][1](#page-1-0) .

# II. RELATED WORK

Learning-based locomotion has evolved along two main lines. In one, a learned module proposes high-level commands such as footholds or body motion that a model-based controller then tracks; examples include RLOC and hierarchical vision-in-the-loop architectures that marry learned planners with Whole Body Controllers or Model Predictive Control for tracking [10], [11]. In the other, policies act end-to-end on the robot—commanding torques or joint targets directly, and can discover strategies that hand-crafted pipelines might miss [4], [8]. While hierarchical schemes benefit from explicit feasibility handling in the low-level controller, they rely on accurate modeling and foothold execution; direct policies reduce hand-engineering but require careful reward design and curricula for stability and transfer.

Perception choices strongly affect both families. Early "blind" approaches (proprioception only) showed surprising robustness but lack anticipation of obstacles [4], [5]. Egocentric depth cameras enable end-to-end perception and have demonstrated stair and gap traversal, yet their limited field of view and sensor noise burden the policy with temporal memory, especially for hind-leg terrain [6]. Other methods depend on globally consistent elevation maps or multi-sensor rigs; these increase foresight but demand careful calibration and accurate global pose estimation, which can be brittle in practice [7]. A robot-centric *local heightmap* offers a compact, task-aligned alternative that captures nearby terrain geometry relevant to all legs without global consistency assumptions [5];

several recent systems, including ours, adopt this representation to couple perception more tightly to control.

A parallel thread injects structure via gait priors. Phase-augmented controllers specify foot or joint targets as functions of a per-leg phase and track them with IK/PD controllers, improving stability but coupling the policy to morphology and introducing action-space bias [4], [8]. Central Pattern Generators (CPG)-based methods similarly embed oscillators and let RL modulate their parameters, inheriting the same limitations [12]. An alternative is to encode gait regularity in the *objective* rather than in the actions: phase-guided reward shaping encourages desired swing/stance timing and foot clearance while leaving the policy free to decide the final commands [13]. This shift reduces inductive bias and eases deployment across platforms with different kinematics.

The training substrate also varies. GPU-accelerated simulators such as Isaac Gym have popularized massively parallel data collection for on-policy RL [9], while the MuJoCo/MJX stack offers accurate contact dynamics with a lightweight, accessible toolchain and good throughput for perception-aware policies. These choices interact with observation design and reward shaping: compact robot-centric inputs and structured, but not action-constraining, objectives typically reduce training instabilities and simplify transfer.

In this landscape, PGTT aligns with direct joint-space control but differs in how structure is injected: it uses a robot-centric heightmap for perception and enforces a *terrain-adaptive, phase-guided prior purely through reward shaping*, avoiding oscillators and IK. This design aims to retain the benefits of rhythmic organization while minimizing action-space constraints, thereby reducing inductive bias and supporting morphology-agnostic deployment relative to oscillator/IK-conditioned policies [4], [8], [13].

## III. PHASE-GUIDED TERRAIN TRAVERSAL

At a high level, Phase-Guided Terrain Traversal (PGTT) combines three ideas (Fig. [4\)](#page-3-0): (i) a compact perception module that encodes terrain as a robot-centric heightmap derived online from LiDAR measurements, (ii) phase variables and reward function that provide rhythmic structure without constraining the action space, and (iii) an asymmetric actor–critic architecture trained with PPO in GPU-accelerated MuJoCo (MJX) environments.

## *A. Problem Formulation*

We model legged locomotion as an infinite-horizon partially observable Markov decision process (POMDP)

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, P, \Omega, r, \gamma, \rho_0),$$

where s<sup>t</sup> ∈S is the full state, a<sup>t</sup> ∈A the action, and o<sup>t</sup> ∈O a partial observation. The transition kernel is P(st+1 |st,at), the observation model (sensor and preprocessing pipeline) is Ω(o<sup>t</sup> |st), r :S ×A→R is the reward, γ ∈[0,1) the discount factor, and ρ<sup>0</sup> the initial-state distribution. In our setting, o<sup>t</sup> comprises proprioception and a robot-centric heightmap derived online from LiDAR, while s<sup>t</sup> additionally includes

<span id="page-1-0"></span><sup>1</sup>Our code is available at [https://github.com/NtagkasAlex/](https://github.com/NtagkasAlex/phase_guided_terrain_traversal) [phase\\_guided\\_terrain\\_traversal](https://github.com/NtagkasAlex/phase_guided_terrain_traversal).

<span id="page-2-1"></span>![](_page_2_Figure_0.jpeg)

Fig. 2: Simulation snapshots of PGTT. Left: Go2 on stairs with projected front-foot trajectories (red). Middle: Go2 traversing discrete obstacles. Right: ANYmal C on stairs.

privileged quantities used only during training. A stochastic policy πθ(a<sup>t</sup> |ot) maximizes the discounted return

$$J(\pi_{\theta}) = \mathbb{E} \underset{\substack{s_0 \sim \rho_0 \\ s_{t+1} \sim P(\cdot|s_t, a_t)}}{\underset{s_{t+1}}{s_0 \sim \rho_0}} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right]. \tag{1}$$

## *B. Robot-Centric Heightmap Representation*

Perception in PGTT relies on a compact representation of the terrain in the form of a *robot-centric heightmap*. Unlike global elevation maps that require pose estimation and multi-sensor calibration, the heightmap is anchored to the robot's body frame and updated in real time from onboard LiDAR or using ground truth information in simulation. We detail how to get the heightmap from onboard LiDAR measurements in Sec. [IV-F.](#page-6-0)

In particular, we create an N ×M grid of equally spaced points around the robot (Fig. [3\)](#page-2-0), and pass it as input to the policy. Because the representation is local and robot-centric, it captures the geometry relevant to both front and hind legs without requiring global localization.

<span id="page-2-0"></span>![](_page_2_Picture_7.jpeg)

Fig. 3: PGTT's robot-centric heightmap.

## *C. Encoding Rhythmic Structure*

Following prior work that leverages rhythmic structure [8], [13], [14], we define for each leg i a periodic phase variable ϕi,t∈[0,2π), interpreted as *contact* for ϕi,t∈[0,π) and *swing* for ϕi,t∈[π,2π). The phase advances with a base frequency f and over time t as follows:

$$\phi_{i,t} = (\phi_{i,0} + 2\pi f t) \bmod 2\pi,$$

where ϕi,<sup>0</sup> sets the inter-leg offsets (gait).

In PGTT, {ϕi,t} provides *structure* but does not constrain the action space: it is used *only* to shape the reward (see Sec. [III-E\)](#page-3-1) rather than to prescribe joint targets via oscillators

or IK, a design that reduces inductive bias during learning while preserving morphology-agnostic deployment.

## *D. Asymmetric Actor-Critic Learning*

Teacher–student distillation is a common recipe for locomotion with partial observations [8], [14]: a teacher is trained with full-state (privileged) inputs and a student subsequently imitates it from observations. However, because the student is constrained to match the teacher's demonstrations, the resulting policy can inherit suboptimality and distribution mismatch from the teacher's occupancy measure. We therefore adopt *asymmetric actor–critic* [15], where the actor πθ(a<sup>t</sup> | ot) conditions only on observations while the critic is trained with privileged state st; this retains the benefits of privileged information for value estimation without constraining the learned behavior by imitation [16].

Action Space: The action space is a 12 × 1 vector, at, corresponding to the desired joint angle of the robot. To facilitate learning, we train the policy to infer the desired joint angle around the robot's stand still pose. Hence, the robot's desired joint angles are computed as

$$q_{des} = q_{def} + ka_t, (2)$$

where k is a constant *action scale* parameter.

Observation Space: The observation space ot, which is passed to the policy network πθ(at|ot), consists of mainly proprioceptive and exteroceptive measurements. To encode the leg phase, we use cos(ϕ),sin(ϕ) instead of ϕ=[ϕ0,ϕ1,ϕ2,ϕ3], which is a smooth and unique representation for the angle [8].

$$o_t = [\omega_t \ g_t \ q_t \ \dot{q}_t \cos(\phi) \sin(\phi) \ h_t \ f \ a_{t-1} \ v_{cmd}]^T,$$
 (3)

where ωt,gt,qt,q˙t,cos(ϕ),sin(ϕ),ht,f,at−<sup>1</sup> and vcmd are the body angular velocity, the gravity vector expressed in the local frame, the joint angles, the joint velocities, the phase representation, the flattened height-scans of the terrain, the base frequency, the last action and the command.

Value Network: The value network is trained to output an estimation of the true state value, V (st). Unlike the policy the state s<sup>t</sup> contains privileged information

$$s_t = [o_t \ v_t \ ]^T, \tag{4}$$

where v<sup>t</sup> is the linear velocity in the local frame. Linear velocity is critical because it correlates strongly with the main objective-track commanded velocity- and thus with the value function output.

<span id="page-3-0"></span>Fig. 4: PGTT combines curriculum learning, a robot-centric heightmap, reward shaping through Hermite splines, asymmetric actor-critic learning, and low-level PD controllers for effective perceptive legged locomotion.

#### <span id="page-3-1"></span>E. Phase-Guided Reward Function

Reward design is central to legged locomotion with reinforcement learning. Most existing approaches combine a forward-velocity tracking term with a set of penalties (slip, foot clearance) to promote stable gaits. While effective, these reward structures often require extensive manual tuning and are usually combined with oscillators or IK-based controllers.

PGTT pursues a different route: we aim to generate phase-guided swing trajectories *without* inverse kinematics. The phase prior influences learning only through the reward, which reduces the number of hand-tuned terms and avoids constraining the policy. The core idea is to use *cubic Hermite splines* to define smooth foot trajectories conditioned on a per-leg phase variable and local terrain information.

We denote by  $p_{f,z,i}$  the z-axis (height) position of foot i in the hip-joint frame, and by  $p_{w,f,z,i}$  the corresponding position in the world frame. Let  $d_b$  be the nominal foot height in stance (default configuration) and  $d_s$  the nominal swing apex (see Fig. 5). To adapt the trajectory to terrain, we compute local statistics around each leg:  $H_{\max,i}$  and  $H_{\min,i}$  are the maximum and minimum terrain heights in the world frame, and  $\delta H_i = H_{\max,i} - H_{\min,i}$  is added to the swing trajectory to guarantee obstacle clearance.

<span id="page-3-2"></span>![](_page_3_Picture_6.jpeg)

Fig. 5: Distances relative to the hip-joint frame and world frame. The *black* leg is the nominal stance, the *dashed* line a possible swing trajectory, and the *red* leg a random leg configuration.

Formally, a cubic Hermite spline is defined by start and end positions  $p_0, p_1$ , tangents  $m_0, m_1$ , and duration T. For

 $t \in [0,T]$ , the trajectory is

$$P(t) = c_0 + c_1 t + c_2 t^2 + c_3 t^3,$$

$$c_0 = p_0, \quad c_1 = m_0,$$

$$c_2 = \frac{3}{T^2} (p_1 - p_0) - \frac{2}{T} m_0 - \frac{1}{T} m_1,$$

$$c_3 = -\frac{2}{T^3} (p_1 - p_0) + \frac{1}{T^2} (m_0 + m_1).$$
(5)

We divide each leg trajectory into three phases (parameterized by  $\phi_{i,t}$ ):

- Stance: foot remains at  $d_b$  until  $\phi_{i,t} = T_{\text{stance}}$ , where  $T_{\text{stance}} = 2\pi p_{\text{stance}}$  and  $p_{\text{stance}}$  is the stance ratio.
- Swing up: spline  $P_{su}$  with parameters  $(d_b, d_s + \delta H_i, 0, 0, T_{\text{swing}})$ , duration  $T_{\text{swing}} = 2\pi (1 p_{\text{stance}})/2$ .
- Swing down: spline  $P_{sd}$  with parameters  $(d_s + \delta H_i, d_b, 0, 0, T_{\text{swing}})$ , starting at  $\phi_{i,t} = T_{\text{peak}} = 2\pi (1 + p_{\text{stance}})/2$ .

The desired z-position of foot i at phase  $\phi_{i,t}$  is then

$$p_{f,z,i}^{\text{des}}(\phi_{i,t},h_t) = \begin{cases} d_b, & 0 \leq \phi_{i,t} < T_{\text{stance}}, \\ P_{su,i}(\phi_{i,t} - T_{\text{stance}},h_t), & T_{\text{stance}} \leq \phi_{i,t} < T_{\text{peak}}, \\ P_{sd,i}(\phi_{i,t} - T_{\text{peak}},h_t), & T_{\text{peak}} \leq \phi_{i,t} < 2\pi. \end{cases}$$
 (6)

This compact definition provides structured yet terrain-adaptive trajectories that encourage swing clearance while leaving the policy free to discover joint-space behaviors. Reward terms penalize deviation from  $p_{f,z,i}^{\mathrm{des}}$  and swing-phase ground contacts, thereby shaping gait without explicit action-space constraints.

In practice, the overall reward is the sum of a small number of terms. Apart from the task-specific rewards (e.g. linear velocity tracking in our case), the central *positive* term encourages each foot to follow its terrain-adaptive, phase-guided trajectory:

$$r_{\text{phase}} = \sum_{i \in \text{feet}} \exp\left(-\frac{\left(p_{f,z,i}^{\text{des}}(\phi_{i,t},h) - p_{f,z,i}\right)^2}{\sigma_f}\right). \tag{7}$$

To discourage premature contacts during swing, we include a *negative* penalty:

$$r_{\text{contact}} = \sum_{i \in \text{feet}} \mathbb{1}_{\pi \le \phi_{i,t} < 2\pi} c_i, \tag{8}$$

where  $c_i = 1$  if foot i is in ground contact and 0 otherwise. This term penalizes collisions when the phase variable indicates that the leg should be swinging.

The final reward for the RL objective is a sum of many subrewards, and can be found along with their weights in Table [I.](#page-5-0)

# *F. Terrain Generation*

To enable robust legged locomotion, we train policies in stair-like environments that capture the structure of indoor stairs while generalizing to boxy obstacles and irregular terrain. We generate these environments using the *Wave Function Collapse* (WFC) algorithm, a constraint-satisfaction procedure originally introduced in [17]. WFC treats the environment as a grid where each cell can take on values (tiles) consistent with local adjacency rules. By iteratively "observing" one cell and propagating its constraints to neighbors, the algorithm produces diverse yet feasible layouts.

Rather than primitive geometric tiles, we define higherlevel architectural units (straight stair segments, corners, and flat floor tiles) as the building blocks. This representation encourages structural realism while still allowing variability. Each environment is represented as a two-dimensional grid of size (2N+1)×(2N+1), with the robot always spawning on the central tile at position (N,N). Although the tiles are three-dimensional structures in simulation, the grid representation simplifies WFC's operation while maintaining spatial consistency.

![](_page_4_Picture_4.jpeg)

Fig. 6: Example of a procedurally generated stair-like environment (w = 0.1, h= 0.08, n= 8).

## <span id="page-4-0"></span>*G. Curriculum Learning*

Direct training on highly irregular or steep terrains can impede convergence and result in brittle behaviors. To mitigate this, we adopt a staged curriculum in which terrain difficulty increases progressively. The curriculum comprises four levels of stair-like environments:

- Level 1: nearly flat steps with minimal elevation changes,
- Level 2: moderate step heights,
- Level 3: tall obstacles requiring clear swing trajectories,
- Level 4: the highest steps the robot can safely traverse, designed to fully test gait robustness.

Training begins on the easiest terrain and advances only once the agent demonstrates reliable performance at each

level. By mastering balance and stepping on low stairs before tackling larger obstacles, the policy gradually acquires the necessary stability and clearance behaviors. This staged progression improves sample efficiency and results in policies that transfer more reliably to challenging real-world scenarios.

# *H. Sim-to-Real Transfer*

To improve robustness and bridge the sim-to-real gap, we apply extensive domain randomization during training. More specifically:

- Sensor noise: Gaussian noise is injected into all components of the observation vector o<sup>t</sup> to mimic measurement uncertainty and encourage robustness to perception artifacts.
- Robot properties: We randomize key physical parameters such as link and torso masses, default joint positions, motor gains (kp, kd), and actuator friction, mitigating sensitivity to modeling errors.
- Environment properties: We vary terrain and stair friction coefficients to account for diverse contact conditions in the real world.

This randomization forces the policy to generalize across a wide range of conditions, improving stability and survivability in hardware deployment.

## *I. Training Details*

We train legged locomotion policies in the MuJoCo physics simulator, using its GPU-accelerated JAX branch (MJX) to achieve high-throughput simulation. On-policy methods such as PPO benefit directly from MJX's vectorized, GPU-friendly operations [18]. Compared to the widely used Isaac Gym [9], MuJoCo/MJX requires significantly less computational power, making large-scale training of perceptive locomotion policies more accessible.

Our pipeline combines procedurally generated stair-like terrains, GPU-accelerated heightmap extraction, curriculum learning, and domain randomization. MuJoCo provides accurate contact dynamics, while MJX allows the full physics computation to run on the GPU. On top of MJX, we leverage BRAX [19], a lightweight differentiable physics engine built in JAX [20], which provides efficient hardware acceleration and clean integration with modern reinforcement learning algorithms. The policy is optimized using Proximal Policy Optimization (PPO) [21], a widely adopted on-policy algorithm.

## IV. EXPERIMENTAL SETUPS AND RESULTS

In this section we will present the results of the proposed policy in simulation and the real world and compare them with the baseline policies in terms of several metrics. All policies were trained on a workstation equipped with an Intel Core i9- 14900K CPU and a single NVIDIA GeForce RTX 3080 GPU. Training used a physics-integration time step of dt= 0.005s. During deployment, in both Sim2Sim and Sim2Real transfers, control commands are issued at 50 Hz (i.e., every 0.02 s).

<span id="page-5-0"></span>TABLE I: Reward functions. First section contains common tasks, second section the rewards for *PGTT*, third section the *MassLoco* method rewards and fourth section the *Wild* method rewards. The indicator function  $\mathbb{1}_c$  obtains the value 1 if c is true and is 0 otherwise.

| Reward                  | Equation $(r_i)$                                                                                                                 | Weight $(w_i)$      |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------|---------------------|
| Lin. velocity tracking  | $\exp\left(-\frac{(\mathbf{v}_{xy}^{\mathrm{cmd}} - \mathbf{v}_{xy})^2}{\sigma_v}\right)$                                        | 1.0                 |
| Ang. velocity tracking  | $\exp\left(-\frac{(\omega_z^{\text{cmd}} - \omega_z)^2}{\sigma_v}\right)'$                                                       | 0.5                 |
| Linear velocity $(z)$   | $v_z^2$                                                                                                                          | -2.0                |
| Angular velocity $(xy)$ | $\begin{array}{c} v_{\tilde{2}}^2 \ \omega_{xy}^2 \  \boldsymbol{g} ^2 \end{array}$                                              | -0.05               |
| Orientation             | $ g ^2$                                                                                                                          | -0.2                |
| Termination             | $\mathbb{1}_{t < T}$                                                                                                             | -1.0                |
| Joint power             | $ \tau  \dot{\theta} $                                                                                                           | $-2 \times 10^{-5}$ |
| Action rate             | $(a_t - a_{t-1})^2$                                                                                                              | -0.01               |
| Joint limits            | $1_{q_i>q_{max}  q_i< q_{min} }$                                                                                                 | -1.0                |
| Default pose            | $\begin{array}{l} \mathbb{1}_{q_i>q_{max}\mid\mid q_i< q_{min}} \\ \sum_{i \in \text{foot}} (q-q_{def}) \cdot w_i \end{array}$   | -0.5                |
| Joint torques           | $ \tau ^2$                                                                                                                       | 0.001               |
| PGTT                    |                                                                                                                                  |                     |
| Foot phase              | $\sum_{i \in \text{foot}} \exp \left( -\frac{\left(p_{f,z,i}^{\text{des}}(\phi_{i,t},h) - p_{f,z,i}\right)^2}{\sigma_f} \right)$ | 0.5                 |
| Foot contact            | $\sum_{i \in \text{foot}} \mathbb{1}_{\pi \leq \phi_{i,t} < 2\pi} c_i$                                                           | -2.0                |
| MassLoco                |                                                                                                                                  |                     |
| Foot clearance          | $\sum_{i \in \text{foot}} (p_{w,f,z,i}^{\text{des}} - p_{w,f,z,i})^2 \cdot   v_{f,xy,i}  _2$                                     | -0.5                |
| Foot slip               | $\sum_{i \in \text{foot}} (\ v_{f,xy,i}\ _2 \cdot c_i)$                                                                          | 0.1                 |
| Feet air time           | $1_{\ v_{cmd}\ _2 > 0.01} \sum_{i \in \text{foot}} (t_{i,air} - 0.5)$                                                            | 1.0                 |
| Stand still             | $1_{\ v_{cmd}\ _2 < 0.01} (q - q_{def})$                                                                                         | 0.5                 |
| Wild                    |                                                                                                                                  |                     |
| Foot clearance          | $\sum_{i \in \text{foot}} \mathbb{1}_{\pi \leq \phi_{i,t} < 2\pi} \mathbb{1}_{p_{w,f,z,i} \geq H_{max,i}}$                       | 0.1                 |
| Foot slip               | $\sum_{i \in \text{foot}} (\ v_{f,xy,i}\ _2 \cdot c_i)$                                                                          | 0.1                 |

#### A. Baselines

We select baseline methods that are both relevant to our problem and representative of existing approaches to enable a fair comparison with our method. To evaluate whether locomotion without fixed gait scheduling can yield more efficient behaviors, we include *MassLoco* [22], including rewards inspired by Margolis et al. [23] to encourage more natural walking patterns. On the other hand, when considering a state-of-the-art method that leverages gait priors, we compare against *Wild* [8]. We did not include Visual CPG-RL [24], since, although its framework is similar to Wild, it is not trained or evaluated on stairs or obstacle traversal, and is therefore considered less relevant for our study.

Table I summarizes the rewards used by our method and the baselines. We include the ordinary rewards for velocity tracking (typically  $\sigma_v=0.25$ ) and some regularization rewards to maintain balance, avoid early termination and excessively large stress of the joints. These rewards were tuned individually for each method to achieve optimal performance. We also include rewards to avoid joint limits, where  $q_{max}, q_{min}$  are soft limits for the joints of the robot, and a reward to penalize deviations from the default joint position with different weights for each joint.

For PGTT two additional rewards are necessary: foot phase is responsible for tracking the desired leg trajectories (with a small  $\sigma_f = 0.05$ ), and foot contact penalizes contacts at swing phase. In contrast, the MassLoco method requires four additional reward terms. Foot clearance promotes high clearance strides, since it penalizes deviation from the desired swing height when the feet move in the xy plane (non-zero velocity), foot slip penalizes contacts when feet move in the xy plane, feet air time promotes long strides when a

command is given and stand still promotes maintaining the default configuration when no command is given. The Wild method uses the same foot slip reward as MassLoco, but its foot clearance reward specifically encourages swings that rise higher than the surrounding obstacles.

#### B. Implementation Details

At the start of each episode we draw a constant command  $v_{cmd} = [v_x \ v_y \ \omega_z]^T$  uniformly from the intervals  $u_{min} = (-1,-1,-1)$  and  $u_{max} = (1,1,1)$ . To expose the policy to discontinuous command changes, we resample  $v_{cmd}$  once per episode at a random time-step. Additionally, we sample frequencies  $f \sim U[1,3]$  to create diverse gaits in terms of timings.

The actor and critic networks are modeled as multilayer perceptrons (MLPs) with hidden layer sizes of 512, 256, and 128. Our choice of using an MLP with elevation maps is justified, as prior work has shown that memory mechanisms are not required for this modality, with both MLP and LSTM architectures achieving comparable reconstruction performance [25]. Episodes are terminated early if the robot turns upside down to further accelerating training.

#### C. Curriculum Learning

We employ Curriculum Learning to structure the training process as detailed in Sec. III-G. At all levels we use a grid size of 5 (2N+1=5), step width is sampled as  $w \sim U[0.3,0.45]$ , number of steps are sample as  $n \sim U[2,4]$ ; the only difference is the height, where in level 1 we sample from 1cm to 3cm , in level 2 from 1cm to 7cm, for level 3 we sample from 1cm to 10cm and fore level 4 from 1cm to 13cm. These choices allow for efficient learning and overcoming progressively higher obstacles, while retaining the ability to traverse smaller ones.

Determining when the agent has successfully completed a level, however, is non-trivial. To address this, we utilize a velocity tracking metric to assess level completion. Specifically, during each evaluation step, using  $n_{eval}$  agents, we compute the cumulative velocity reward as follows:

$$m_v = \frac{1}{n_{eval}T} \frac{1}{w_v} \sum_{e=1}^{n_{eval}} \sum_{t=0}^{T-1} w_v e^{-\frac{(v_t - v_{cmd})^2}{\sigma}}$$
(9)

$$m_{\omega} = \frac{1}{n_{eval}T} \frac{1}{w_{\omega}} \sum_{e=1}^{n_{eval}T-1} \sum_{t=0}^{T-1} w_{\omega} e^{-\frac{(\omega_t - \omega_{cmd})^2}{\sigma}}$$
 (10)

Therefore, we define success of a level if  $m_v, m_\omega \ge p$ , where  $p \in [0,1)$ . We set p=0.65. Additionally, we will declare convergence only when the episode reward  $R_t$  has stabilized, that is:

$$\frac{|R_t - R_{t-1}|}{R_{t-1}} < \epsilon \tag{11}$$

#### D. Metrics

We compare our method PGTT with the two aforementioned baseline MassLoco and Wild in terms of linear and angular velocity tracking and success rate. The

<span id="page-6-1"></span>![](_page_6_Figure_0.jpeg)

Fig. 7: Comparison of PGTT with baseline methods MassLoco and Wild across three metrics: linear velocity tracking, angular velocity tracking and success rate. Solid lines show the median over 5 different training seeds and the shaded regions are the regions between the 25-th and 75-th percentiles.

<span id="page-6-2"></span>TABLE II: Evaluation metrics when the robot traverses discrete obstacles of varying height from 2cm to 9cm. Success rate, normalized body linear velocity error  $\bar{v}$ , and normalized body angular velocity error  $\bar{\omega}$  for 1000 quadrupeds. Results are (**median**, 25th percentile, 75th percentile) over 5 training seeds.

| Method   | Success<br>Rate              | $\bar{v}$                    | $\bar{\omega}$               |
|----------|------------------------------|------------------------------|------------------------------|
| PGTT     | ( <b>0.848</b> ,0.842,0.855) | ( <b>0.965</b> ,0.958,0.972) | (0.991,0.986,0.994)          |
| MassLoco | ( <b>0.702</b> ,0.659,0.711) | ( <b>0.983</b> ,0.939,0.986) | ( <b>0.903</b> ,0.863,0.904) |
| Wild     | ( <b>0.756</b> ,0.756,0.769) | (0.998,0.998,1.000)          | ( <b>0.935</b> ,0.935,0.941) |

first two are defined as in Table I. Success rate (SR) is defined as follows *Gangapurwala et al.* [26]:

$$SR = 1 - \frac{N_e}{N_T},\tag{12}$$

with  $N_e$  referring to the number of rollouts that terminated early due to a prohibited behavior and  $N_T$  being the total number of rollouts. Using  $N_T=1000$ , we randomize the base linear and angular velocity command with  $0.7 \cdot v^{\rm max}$  from the one used during training.

#### E. Simulation Results

We compared the three methods in generated stair environments with obstacle heights ranging from 2cm to 9cm. While the robot can traverse obstacles up to 12cm, the most stable behaviors were observed at heights between 7cm and 9cm. We would like to compare the methods when perturbations are applied to the robot torso towards any direction, to validate the robustness in realistic scenarios. We apply perturbations of uniformly sampled magnitude between 7.5 to 30 N and we also sample durations and wait times between consecutive perturbations. We replicate the whole training pipeline over 5 different seeds. All metrics excluding the success rate are normalized with respect the the maximum value. The results reveal several clear trends (Fig. 7): PGTT and Wild exhibit very similar commanded velocity tracking, whereas MassLoco lags behind. PGTT achieves the highest success rate, outperforming the second-best method, Wild, by 7.5% on average.

Similar behavior is observed when evaluating the three methods in environments with discrete obstacles, with PGTT achieving the highest success rate; **9% higher than the second-best method**. Additionally, both angular and linear velocity tracking are very similar across all methods (Table II).

In terms of convergence of the policy learning, PGTT and Wild are mostly equivalent, whereas MassLoco is considerably less sample-efficient, needing about twice as many steps in the first curriculum level (Figure 8). In terms of wall-time, the mean completion time (averaged over five training seeds) across all four levels are 195, 198, and 239 minutes for PGTT, Wild, and MassLoco, respectively.

<span id="page-6-3"></span>![](_page_6_Figure_11.jpeg)

Fig. 8: Training curves. We report only for Level 1 of our curriculum learning, since this is the level with the biggest differences. *Solid lines* show the median over 5 different training seeds and the shaded regions are the regions between the 25-th and 75-th percentiles.

**ANYmal C Experiments:** We also applied our method on the same task with the ANYmal C robot. Our preliminary experiments showcase that PGTT is able to generate walking behaviors without even changing the hyper-parameters (see Fig. 2 and the supplementary video).

#### <span id="page-6-0"></span>F. Real-World Deployment

We evaluate the Sim2Real capabilities of our method on a Unitree Go2 quadruped. For perception, a L1 LiDAR is fused with IMU data using Point-LIO [27], a tightly coupled LiDAR–Inertial Odometry framework. The resulting odometry and transformed point cloud are used to construct a robot-centric elevation grid map [28], where each cell (i,j) stores a mean height  $\hat{h}ij$  and variance  $\sigma ij^2$  to represent terrain uncertainty. Since raw LiDAR maps often contain holes (NaN values) that can destabilize the policy, we apply a median-fill filter that in-paints only small gaps (below radius  $r_{\text{hole}}$ ) surrounded by reliable data, while leaving larger unknown regions untouched.

To provide real-time input to the policy, we extract a robot-centric heightmap by sampling an  $11 \times 9$  grid within a  $1.1m \times 0.9m$  area centered on the robot. Although accuracy

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Fig. 9: Real world experiments. The bottom left image shows the gridmap and the bottom right shows the odometry.

is bounded by grid resolution, the domain randomization used during training makes the policy robust to such imperfections. The locomotion policy executes at 50Hz, producing joint targets that are translated into torques through a lightweight PD controller (k<sup>p</sup> = 60, k<sup>d</sup> = 3) before being applied by the Go2's onboard low-level controller.

Our experiments showcase that policies trained with the PGTT effectively transfer to the real-world (Fig. [9\)](#page-7-0), and the robot is able to walk both on static stair and discrete obstacles environments, and withstand real-life perturbations. The supplementary video showcases such examples.

## V. CONCLUSION

In this work, we introduced Phase-Guided Terrain Traversal (PGTT), a perception-aware locomotion framework that integrates local heightmap perception, reinforcement learning, and terrain-adaptive gait priors to achieve robust and efficient terrain traversal. Our results demonstrate that PGTT increases traversal success rates by 7.5% and converges to a strong policy approximately 2× faster compared to state-of-the-art baselines. Moreover, it generalizes across robot platforms without relying on inverse kinematics and transfers successfully to real hardware, as demonstrated by reliable deployment on a Unitree Go2. A key strength of PGTT is its reliance on the lightweight MuJoCo simulation stack, which allows perception-aware locomotion policies to be developed and trained on affordable hardware such as a single consumer-grade GPU, thus lowering the barrier to entry for this line of research.

At the same time, our approach has limitations. The L1 LiDAR used in our implementation outputs only 21,600 points/s, considerably fewer than higher-end sensors (≥200,000 points/s). At higher locomotion speeds, this leads to sparser measurements and less frequent map updates, restricting our experiments to a maximum speed of 0.4m/s. This issue, however, is tied to sensing hardware and could be readily addressed with improved sensors. Another limitation is that we have not evaluated the energy efficiency of PGTT in real-world deployments; understanding its effect on power consumption will be important for scaling to long-duration or field applications.

Overall, PGTT provides an accessible and effective foundation for advancing agile, robust, and affordable legged locomotion in real-world environments, empowering researchers and laboratories without extensive computational resources to contribute to this field.

## REFERENCES

- [1] M. Hutter, C. Gehring, A. Lauber, *et al.*, "Anymal a highly mobile and dynamic quadrupedal robot," *IEEE/RSJ IROS*, 2017.
- [2] S. Kuindersma, F. Permenter, and R. Tedrake, "Optimization-based locomotion planning, estimation, and control design for the atlas humanoid robot," *Autonomous Robots*, vol. 40, pp. 429–455, 2016.
- [3] A. Kumar, K. Jatavallabhula, *et al.*, "Rma: Rapid motor adaptation for legged robots," *Robotics: Science and Systems (RSS)*, 2021.
- [4] J. Lee, J. Hwangbo, *et al.*, "Learning quadrupedal locomotion over challenging terrain," *Science Robotics*, 2020.
- [5] Y. Duan, M. Zhang, *et al.*, "Learning a bipedal walking policy via reinforcement learning and gait library," in *Conference on Robot Learning (CoRL)*, 2021.
- [6] A. Agarwal, A. Kumar, *et al.*, "Legged locomotion in challenging terrains using egocentric vision," in *Robotics: Science and Systems (RSS)*, 2022.
- [7] T. Miki, J. Lee, and M. Hutter, "Expanding the horizon: Perceptionaware mpc for agile vision-based legged locomotion," in *Conference on Robot Learning (CoRL)*, 2022.
- [8] T. Miki, J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, "Learning robust perceptive locomotion for quadrupedal robots in the wild," *Science Robotics*, vol. 7, no. 62, p. eabk2822, 2022.
- [9] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State, "Isaac gym: High performance gpu-based physics simulation for robot learning," 2021.
- [10] S. Gangapurwala, M. Geisert, R. Orsolino, M. Fallon, and I. Havoutis, "Rloc: Terrain-aware legged locomotion using reinforcement learning and optimal control," *IEEE Transactions on Robotics*, vol. 38, no. 5, pp. 2908–2927, 2022.
- [11] W. Yu, D. Jain, A. Escontrela, A. Iscen, P. Xu, E. Coumans, S. Ha, J. Tan, and T. Zhang, "Visual-locomotion: Learning to walk on complex terrains with vision," in *5th Conference on Robot Learning (CoRL)*, vol. 164 of *Proceedings of Machine Learning Research*, pp. 1291–1302, 2022.
- [12] G. Bellegarda and A. J. Ijspeert, "Cpg-rl: Learning central pattern generators for quadruped locomotion," *IEEE Robotics and Automation Letters*, vol. 7, no. 4, pp. 12547–12554, 2022.
- [13] Y. Shao, Y. Jin, X. Liu, W. He, H. Wang, and W. Yang, "Learning free gait transition for quadruped robots via phase-guided controller," *IEEE Robotics and Automation Letters*, vol. 7, p. 1230–1237, Apr. 2022.
- [14] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, "Learning quadrupedal locomotion over challenging terrain," *Science Robotics*, vol. 5, no. 47, p. eabc5986, 2020.
- [15] L. Pinto, M. Andrychowicz, P. Welinder, W. Zaremba, and P. Abbeel, "Asymmetric actor critic for image-based robot learning," 2017.
- [16] I. M. A. Nahrendra, B. Yu, and H. Myung, "Dreamwaq: Learning robust quadrupedal locomotion with implicit terrain imagination via deep reinforcement learning," 2023.
- [17] M. Gumin, "Wave Function Collapse Algorithm," Sept. 2016.
- [18] K. Zakka, B. Tabanpour, Q. Liao, M. Haiderbhai, S. Holt, J. Y. Luo, A. Allshire, E. Frey, K. Sreenath, L. A. Kahrs, C. Sferrazza, Y. Tassa, and P. Abbeel, "Mujoco playground," 2025.
- [19] C. D. Freeman, E. Frey, A. Raichuk, S. Girgin, I. Mordatch, and O. Bachem, "Brax – a differentiable physics engine for large scale rigid body simulation," 2021.
- [20] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang, "JAX: composable transformations of Python+NumPy programs," 2018.
- [21] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," in *arXiv preprint arXiv:1707.06347*, 2017.
- [22] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, "Learning to walk in minutes using massively parallel deep reinforcement learning," 2022.
- [23] G. B. Margolis and P. Agrawal, "Walk these ways: Tuning robot control for generalization with multiplicity of behavior," 2022.

- [24] G. Bellegarda, M. Shafiee, and A. Ijspeert, "Visual cpg-rl: Learning central pattern generators for visually-guided quadruped locomotion," 2024.
- [25] N. Rudin, J. He, J. Aurand, and M. Hutter, "Parkour in the wild: Learning a general and extensible agile locomotion policy using multi-expert distillation and rl fine-tuning," 2025.
- [26] S. Gangapurwala, L. Campanaro, and I. Havoutis, "Learning lowfrequency motion control for robust and dynamic robot locomotion," 2023.
- [27] D. He, W. Xu, N. Chen, F. Kong, C. Yuan, and F. Zhang, "Point–LIO: Robust high-bandwidth lidar–inertial odometry," *Advanced Intelligent Systems*, vol. 5, no. 7, p. 2200459, 2023.
- [28] P. Fankhauser, M. Bloesch, and M. Hutter, "Probabilistic terrain mapping for mobile robots with uncertain localization," *IEEE Robotics and Automation Letters*, vol. 3, no. 4, pp. 3019–3026, 2018.

---

## Notes

- **Title:** PGTT: Phase-Guided Terrain Traversal for Perceptive Legged Locomotion
- **URL:** https://arxiv.org/pdf/2510.18348
