# Whole-Body End-Effector Pose Tracking

Tifanny Portela1,<sup>2</sup> , Andrei Cramariuc<sup>1</sup> , Mayank Mittal1,<sup>3</sup> and Marco Hutter<sup>1</sup>

*Abstract*— Combining manipulation with the mobility of legged robots is essential for a wide range of robotic applications. However, integrating an arm with a mobile base significantly increases the system's complexity, making precise end-effector control challenging. Existing model-based approaches are often constrained by their modeling assumptions, leading to limited robustness. Meanwhile, recent Reinforcement Learning (RL) implementations restrict the arm's workspace to be in front of the robot or track only the position to obtain decent tracking accuracy. In this work, we address these limitations by introducing a whole-body RL formulation for end-effector pose tracking in a large workspace on rough, unstructured terrains. Our proposed method involves a terrain-aware sampling strategy for the robot's initial configuration and end-effector pose commands, as well as a game-based curriculum to extend the robot's operating range. We validate our approach on the ANYmal quadrupedal robot with a six DoF robotic arm. Through our experiments, we show that the learned controller achieves precise command tracking over a large workspace and adapts across varying terrains such as stairs and slopes. On deployment, it achieves a pose-tracking error of 2.64 cm and 3.64◦ , outperforming existing competitive baselines. The video of our work is available at: **[wholebody-pose-tracking](https://tif-twirl-13.github.io/wholebody-pose-control)**.

# I. INTRODUCTION

Over the past decade, algorithmic advancements have substantially increased legged robots' ability to traverse complex, cluttered environments and human-designed infrastructures, such as stairs and slopes [1]–[3]. Despite these improvements, their practical applicability remains constrained by their limited manipulation capabilities. Most field operations with legged robots involve minimal environmental interactions, such as visual inspections and passive load transportation. Thus, combining a legged robot's mobility with the ability to perform manipulation tasks is critical for enhancing their applications to more real-world scenarios.

Compared to fixed base counterparts, integrating an arm onto a legged mobile platform significantly complicates the controller design because of increased degrees of freedom, redundancy and highly non-linear dynamics. To address this, the research community has mainly explored model-based and learning-based control strategies.

Model-based approaches, such as Model Predictive Control (MPC), have shown precise control on flat terrain by leveraging accurate models of the robot and its environment [4],

Authors are members of <sup>1</sup>Robotic Systems Lab, ETH Zurich. <sup>2</sup>ETH AI center. <sup>3</sup>NVIDIA. Email: tifanny.portela@ai.ethz.ch

This project has received funding from the ETH AI Center and the European Union's Horizon Europe Framework Programme under grant agreement No 101121321 and NCCR automation. This work has been conducted as part of ANYmal Research, a community to advance legged robotics.

![](_page_0_Picture_10.jpeg)

<span id="page-0-0"></span>![](_page_0_Picture_11.jpeg)

Fig. 1: Our whole-body controller demonstrates precise endeffector pose tracking across a variety of challenging terrains, including soft mattresses, stairs and uneven natural ground.

[5]. However, solving MPC's control problem in real-time for legged manipulators often requires the use of simplified models [6], [7], which increases vulnerability to unexpected disturbances such as slipping or unplanned contacts.

In contrast, the learning-based control strategy of Reinforcement Learning (RL) has emerged as a robust alternative, directly learning control policies through interactions in simulation for locomotion [1], [3], [8] and whole-body control [9]. RL enables improved resilience to external disturbances compared to MPC because of environmental variability during training [1], [10], [11]. Research on legged robots, whether using legs [12], [13] or attached arms [9], [14], demonstrate that RL techniques can achieve effective endeffector position tracking across a large workspace through agile whole-body behavior. Although effective in outdoor and slippery terrains, these approaches remain unsuitable beyond flat terrain and lack orientation tracking.

To address these shortcomings, we propose a generalpurpose whole-body controller for legged robots with an attached arm. The controller is designed to provide stable end-effector pose tracking across a large operational space. Our approach includes a terrain-aware sampling strategy for end-effector pose commands, and for the robot's initial configuration to ensure a smooth transition from a locomotion policy to the proposed whole-body controller. Experimental results show that the controller achieves precise tracking across varying terrains, such as stairs and slopes. Our key

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Fig. 2: The training process begins with data collection, where we gather (A) the terrain mesh and a coarse terrain height map, (B) 10000 pre-sampled end-effector pose commands with a fixed base, and (C) base poses and joint angles from a pre-trained locomotion policy to initialize robots. During training, a command from (B) is slightly transformed and checked for collisions with the terrain. If collision-free, it is concatenated with observations and input to the policy; otherwise, a new command is sampled. The policy is trained in simulation with 4000 robots in parallel, outputting joint actions.

contributions are as follows:

- We propose an RL whole-body controller for 6-DoF end-effector pose tracking for quadrupeds with an arm.
- We showcase the learned controller's tracking capabilities over challenging terrains and its robustness when faced with external disturbances.
- We compare our learned controller to model-based controllers and existing RL approaches, showing higher tracking accuracy and enlarged pose reachability, both in simulation and on hardware, reaching a pose tracking accuracy of 2.64 cm and 3.64◦ .

## II. RELATED WORK

# *A. Model-based whole-body control*

Formulating the whole-body planning and control of legged mobile manipulators into a single optimization problem avoids treating the arm as an external disturbance to the base [15], [16] and helps ensure tight coordination between the base and the arm [17], [18]. Belliscoso *et al*. [4] propose an online ZMP-based motion planning framework that relies on an inverse dynamics model to track generated operational-space references. In contrast, Sleiman *et al*. [5] provide a unified MPC formulation for whole-body dynamic locomotion and manipulation planning. Chiu *et al*. [19] further extend this formulation to account for self and environment collisions during the receding-horizon control. To increase the robustness towards external disturbances proactively, Ferrolho *et al*. [20], [21] incorporate a robustness metric into the trajectory optimization to compute offline plans for interactions under worst-case scenario. While these approaches are effective

for end-effector control of legged mobile manipulators, they rely on an available system model and pre-specified contact schedule. These assumptions make model-based approaches vulnerable to large unmodeled disturbances (for instance, a heavy grasped object), slippages, and unknown terrain models.

## *B. Learning-based whole-body control*

More recently, RL has shown to be a powerful alternative to model-based approaches for legged robots, providing robust control policies for locomotion on unstructured terrains [1], [3] and whole-body manipulation [9], [12], [22]. In the context of task-space tracking of a legged mobile manipulator's arm, existing works formulate the tracking problem using different rewards and representations for the end-effector commands.

Ma *et al*. [23] propose a hybrid approach that employs an MPC planner for the arm and RL policy for locomotion. The MPC planner outputs arm joint position and base velocity commands. Piu *et al*. [24] follow a similar approach but train an RL policy for the arm to replace the MPC planner. The RL policy receives the end-effector pose command as a 3D position and Euler angles. Liu *et al*. [25] leverage a hierarchical approach to train a high-level policy that provides commands for an RL policy for the base and inverse kinematics controller for the arm. However, fixing a locomotion policy for the base limits the participation of the legged base in enhancing the workspace of the arm.

Fu *et al*. [9] train a combined RL policy for locomotion and manipulation. For the arm, they sample position commands in spherical coordinates and orientation commands uniformly in SO(3). However, their results are shown only for 3D position commands and on flat terrain. A follow-up comparison in [26]

reports a poor pose tracking accuracy with this work as 22.2cm and 66.22°. Considering the importance of regulating forces during interactions, Portela *et al.* [14] train an RL controller for tracking end-effector 3D position and 3D force simultaneously. However, they also demonstrate their work only on flat terrains. More recently, Ha *et al.* [26] use a 3D Cartesian and 6D rotation representation [27] for the end-effector pose and train a low-level RL policy to track a trajectory of pose commands. While they achieve a position and orientation error of 2.12cm and 3.35°, respectively, the learned controller is mainly demonstrated over reference end-effector motions in front of the robot, which does not necessitate active base (leg) usage.

Our work addresses key issues in existing approaches, including poor pose tracking accuracy, limited workspace, and inability to handle complex terrains. We overcome these challenges by developing a whole-body controller capable of precise end-effector pose tracking across a large workspace and challenging terrains, including stairs.

#### III. METHOD

We train a policy to track end-effector target poses with minimal foot displacement, intended for use alongside a locomotion policy. Figure 2 illustrates the overall training process. We use Isaac Lab [28] as a simulation environment to train our policy and deploy our controller on ALMA [4], which integrates the Anymal D robot from ANYbotics [29] with the Dynaarm from Duatic [30]. The robot has an inertial measurement unit providing body orientation and 18 actuators with joint position encoders. The main control loop and the state estimator are executed at 400 Hz while our policy runs at 50Hz on an onboard computer. While we show results on ALMA, our method remains applicable across different robot embodiments.

## A. Policy Architecture

We use Proximal Policy Optimization (PPO) [31], where both the actor and critic networks are implemented as multi-layer perceptrons with hidden layers of size [512, 256, 128], with ELU as the activation function. The hyperparameters of the PPO algorithm are taken from prior work [32].

#### B. Command sampling

We pre-sample end-effector pose commands for a fixed base by iterating over the six joint angles of the arm, covering their entire range, and recording the end-effector poses that are collision-free in the base frame, as illustrated in Figure 2-B. The final dataset contains 10000 collision-free end-effector poses, with Figure 3A displaying a 250-pose subsample that illustrates the workspace. If these poses were used directly as commands, a simple inverse kinematics solver for the arm would suffice. However, this command sampling strategy would limit the controller's ability to achieve certain end-effector poses that could otherwise be reached by utilizing the entire body of the mobile manipulator. To overcome this limitation, we introduce a random body pose transformation  $T_b^{\Lambda} \in \mathbb{R}^6$ , applied to each pre-sampled command when a new

<span id="page-2-0"></span>![](_page_2_Picture_8.jpeg)

![](_page_2_Picture_9.jpeg)

(A) Initial workspace from sampling arm movements only.

![](_page_2_Picture_11.jpeg)

![](_page_2_Picture_12.jpeg)

(B) Expanded workspace obtained through moving the body and combining it with the initial arm-only workspace. The boundary of the initial workspace is shown in purple.

![](_page_2_Picture_14.jpeg)

![](_page_2_Picture_15.jpeg)

(C) Expanded workspace on stairs, with poses in collision (red) removed based on the coarse terrain height map.

Fig. 3: Front and side views of the workspace with 10000 collision-free end-effector poses (subsampled to 250 and illustrating only positions for readability).

pose is defined. This transformation is sampled within ranges of [-0.2, 0.2] m for the x and y dimensions, [-0.3, 0.1] m for z and,  $[-\pi/6, \pi/6]$  rad for roll, pitch and yaw. This approach ensures that the end-effector pose commands are reachable with minimal base movement. Figure 3B displays a 250-pose subsample of this expanded workspace.

The distribution of the pre-sampled commands is not uniform. We fit five concentric cylinders aligned along the base z-axis. The first cylinder is solid, and each subsequent cylinder has a larger radius with a hollow section. We determine their radii by first calculating the maximum radius in the x-y plane from the 3D positions of all pre-sampled poses and dividing it into five equal parts. The distribution of poses across these cylinders, starting from the inner one, are as follows: 51%, 21%, 13%, 6%, and 2%. To enforce more spatially uniform sampling, we randomly select one of the bins and then sample a pose from within that selected bin.

On flat terrain, this sampling procedure generally yields

feasible commands (a few poses might occasionally intersect with the robot's body due to the added base offset). However, some pose commands can fall below the terrain surface on unstructured terrain like stairs, as illustrated in Figure 3C. To mitigate this, we resample any command where the z-position component falls below the terrain surface height plus an 8 cm margin. We generate a coarse terrain grid map at the start of training to increase training speed and avoid real-time terrain queries. In the x-y plane, this map stores the highest terrain point within a  $20 \times 20$  cm patch centered on each point, as illustrated in Figure 2-A.

Finally, a new command is generated every 4 seconds, and with a 12-second episode length, the robot is exposed to 3 different commands per episode. This setup allows the robot to learn how to reach an end-effector pose from any arm configuration instead of a single-pose episode, where it would always do so from the default arm configuration.

## C. Command definition

The command of the policy is an end-effector pose  $p_{ee} \in \mathbb{SE}(3)$ , typically represented by a separate position and orientation [9], [26]. Separating these components introduces two main challenges. First, defining a rotation representation that is easily learnable is difficult [33]. Second, this separation requires a fixed trade-off between position and orientation rewards, which may not be optimal for all workspace poses.

To avoid these issues, we use a keypoint-based representation similar to that used in [34] for in-hand cube reorientation, which has been shown to improve the ability of the RL algorithm to learn the task at hand. In this formulation, the keypoints represent the vertices of a cube centered on the end-effector's position and aligned with its orientation. While 8 corner points fully define the cube, we use the minimum required of 3 keypoints with direct correspondence between measured and target poses to uniquely and completely define the pose. The side length of the cube is set to 0.3 meters.

## D. Action and Observation Space

The robot's movement is managed through an eighteen-dimensional space ( $a^t \in \mathbb{R}^{18}$ ). This action space controls position targets for a proportional-derivative controller applied to each robot joint. The joints include the legs' thigh, calf, and hip joints and the six arm joints. The position targets are derived as  $\sigma_a a^t + q_{def}$ , where  $\sigma_a = 0.5$  is a scaling factor, and  $q_{def}$  represents the robot's default joint configuration, which corresponds to the robot standing with its arm raised.

The observation, represented as  $o^t$ , relies solely on proprioceptive information. Its elements consist of the gravity vector projected in the base frame  $g_b^t \in \mathbb{R}^3$ , the base linear and angular velocities,  $v_b^t \in \mathbb{R}^6$ , the joint positions,  $q^t \in \mathbb{R}^{18}$  and the previous actions  $a^{t-1} \in \mathbb{R}^{18}$ :

$$o^{t} = [q_{b}^{t}, v_{b}^{t}, q^{t}, a^{t-1}] \in \mathbb{R}^{45} \tag{1}$$

The observation,  $o^t$ , is augmented with the end-effector pose command for the policy input. This command is defined as the positional difference between the current and the target keypoints of the end-effector in the base frame  ${}^bp^{\rm cmd}_{ee} \in \mathbb{R}^9$ ,

where each of the three keypoints provides a 3D position vector, resulting in a 9-dimensional vector in total. The observation does not include explicit terrain information. Nevertheless, as the robot poses are sampled from collision-free configurations (III-G) and the robot remains mostly stationary, it can deduce terrain properties indirectly.

#### E. Rewards

The final reward R is the sum of the task rewards  $R_T$  and penalties  $R_P$ :  $R=R_T+R_P$ . The task reward  $R_T$  can be divided into four subcategories: tracking, progress, feet contact force, and initial leg joint rewards:  $R_T=\omega_1R_t+\omega_2R_p+\omega_3R_f+\omega_4R_q$ , where  $\omega_1=13,\,\omega_2=80,\,\omega_3=0.015$  and  $\omega_4=0.4$ .

**Tracking Reward**  $(R_t)$  is a delayed reward focused on tracking the three keypoints, representing the end-effector pose command, during the last two seconds  $(T_r = 2s)$  of a 4-second command cycle (T = 4s). Delaying the reward emphasizes the importance of being in the correct pose during the final 2 seconds without penalizing the path to getting there. This prevents the unwanted behavior that continuous rewards might encourage, such as passing through the robot's body when transitioning from a pose on one side of the robot to a target on the opposite side.

$$R_t = \begin{cases} \frac{1}{T_r} \sum_{k=0}^{3} e^{\frac{1}{\sigma_t} \|^b p_{ee,k}^{\text{meas}} - b_{ee,k}^{\text{cmd}}\|_2} & \text{if } t > T - T_r \\ 0 & \text{otherwise} \end{cases}$$

Here,  ${}^bp_{ee,k}^{\rm meas}$  and  ${}^bp_{ee,k}^{\rm cmd}$  are the positions of the measured and commanded keypoints in  $\mathbb{R}^3$ , respectively, and  $\sigma_t=0.05$ .

**Progress Reward**  $(R_p)$  addresses the sparsity of the tracking reward and reduces end-effector oscillations by incentivizing steady progress toward the target. It compares the current distance  $d^t \in \mathbb{R}^3$  between the measured and commanded keypoints to the smallest previously recorded distance  $d \in \mathbb{R}^3$ . If  $d^t$  is smaller, the reward is calculated as:

$$R_p = \begin{cases} \frac{1}{3} \sum_{k=0}^{3} (d_k - d_k^t) & \text{if } d^t < d\\ 0 & \text{otherwise} \end{cases}$$

Feet Contact Force Reward  $(R_f)$  encourages the robot to maintain ground contact with all four feet. The reward is non-zero only if all four feet are firmly in contact with the ground. To account for small disturbances and ensure genuine contact, 1 Newton is subtracted from the force on each foot, denoted as  $F_i$ . The reward is calculated as the sum of these adjusted forces:

$$R_f = \sum_{i=1}^{4} \max(F_i - 1, 0)$$

**Initial Leg Joint Reward** ( $R_q$ ) encourages the robot to maintain its leg joints in a configuration close to those sampled from the locomotion policy, as these are known to result in a stable posture, with  $\sigma_q$  defined as 0.05.

$$R_q = \sum_{i=0}^{12} e^{\frac{1}{\sigma_q}(q_i^{\text{init}} - q_i)}$$

**Penalties**  $(R_P)$  penalize joint torques, joint accelerations, action rates and target joint positions above the limits:  $R_P = \omega_5 \|\boldsymbol{\tau}\|^2 + \omega_6 \|\ddot{\boldsymbol{q}}\|^2 + \omega_7 \|\boldsymbol{a}_t - \boldsymbol{a}_{t-1}\|^2 + \omega_8 \|\boldsymbol{q} - \boldsymbol{q}_{\text{lim}}\|_1$ , where  $\omega_5 = -3e^{-5}$ ,  $\omega_6 = -3e^{-6}$ ,  $\omega_7 = -5e^{-2}$  and  $\omega_8 = -1.3$ . Finally, we terminate on knee and base contacts.

## F. Terrains and Curriculum training

The robots are trained in simulation on four procedurally generated terrains: flat, randomly rough, discrete obstacles, and stairs, as defined in [32]. Gradually increasing task difficulty during training has been shown to enhance learning [1], [32], [35]. We employ a terrain curriculum similar to the one proposed in [32], adapted for end-effector pose tracking. If the average position tracking error across the three commands within an episode is under 20 cm when the task reward is active (during the final 2 seconds of each 4-second command), and the average orientation tracking error is under 20°, the terrain difficulty increases after the next reset. The terrain difficulty decreases if the error exceeds 80 cm and 120°. Robots at the highest level are assigned a random level to prevent catastrophic forgetting.

## <span id="page-4-0"></span>G. Initial poses

The whole-body end-effector pose tracking policy proposed in this work does not include locomotion capabilities. This decision is motivated by the observation that very few tasks require simultaneous locomotion and active object manipulation. Therefore, this policy is designed to operate alongside a separate locomotion policy.

To facilitate the learning of a stable leg posture, the initial base pose  $p_b^{init} \in \mathbb{SE}(3)$  and joint angles  $q^{init} \in \mathbb{R}^{18}$  of the robots upon reset are taken from a pre-trained locomotion policy [36]. This policy, designed for the same mobile-legged manipulator using RL, incorporates arm joint positions and velocity measurements in its observation space. It only controls the leg joints, enabling rough terrain locomotion with any arm configuration. This locomotion policy takes a three-dimensional base velocity command as input: linear velocities along the x and y axes and yaw rotation.

Before training, robots are spawned at the center of the terrain and given randomized heading commands in the interval [-1,1] rad s<sup>-1</sup> and linear velocity commands in the interval [-1,1] m s<sup>-1</sup> for a 4-second period, as illustrated in Figure 2-C. After this time, the terrain, the base pose  $p_b$ , and the joint angles q are recorded for robots that remain in stable configurations, defined as those with an angle difference between the gravity vector and their base-aligned projected gravity vector less than 55°. When training begins, the saved terrain and robot configurations are loaded. This process also ensures a smooth transition from the locomotion to the whole-body policy, as omitting this step causes the robot to jump when switching between policies.

# H. Sim-to-Real

When the training starts, we add a mass on the end-effector randomly sampled from the interval [0, 2.0] kg, and the inertia of this rigid body links is scaled by the ratio between the new

<span id="page-4-1"></span>![](_page_4_Figure_9.jpeg)

Fig. 4: Distribution of the position and orientation errors for 10000 end-effector pose commands measured on flat terrain in simulation for four different pose representations.

mass and the original one. Random perturbances are applied to the end-effector, such as an impulse force sampled from the interval [-10,10] N every 3 to 4 seconds, and random pushes on the robot's base simulated as base velocity impulses sampled from the interval [-0.5,0.5] m s<sup>-1</sup> along the x-y dimensions. Random noise is also added to the observations.

## IV. RESULTS AND DISCUSSION

#### A. Simulation experiments

1) Pose representation comparison: To analyze different representations for end-effector pose commands, we evaluate the tracking performance of our keypoint-based pose representation against three other representations from related works: a 3D vector for end-effector position combined with (A) a quaternion [9], (B) with Euler angles [24], or (C) with a 6D vector representation [26]. For each representation, the command is adapted to include both the position and orientation differences expressed in the base frame. Specifically, for (A), we use quaternion multiplication with the inverse of the measured quaternion, we use the difference of Euler angles for (B), and the difference between 6D vectors for (C).

Our tracking reward is adapted to  $R_t = \frac{1}{T_r} e^{\frac{1}{\sigma_t} (\Delta_{\rm pos}^t + \Delta_{\rm rot}^t)}$  for  $t > T - T_r$ , where  $\sigma_t = 0.15$ . The progress reward is given by  $R_p = (\Delta_{\rm pos} - \Delta_{\rm pos}^t) + (\Delta_{\rm rot} - \Delta_{\rm rot}^t)$ , provided that both  $\Delta_{\rm pos}^t < \Delta_{\rm pos}$  and  $\Delta_{\rm rot}^t < \Delta_{\rm rot}$ . The position error  $\Delta_{\rm pos}^t$  is calculated as the norm of the difference in 3D position, while the orientation error  $\Delta_{\rm rot}^t$  is derived from the rotational difference between quaternions. For all pose representations, including (A), (B), and (C), we convert the rotations back to quaternions, multiply one quaternion by the conjugate of the other, and then convert the result to an axis-angle representation to calculate the error.

As shown in Figure 4, the keypoint-based pose representation significantly outperforms the other three, with the 6D representation (C) ranking second with an average tracking error of 16.03 cm and 3.87° larger than ours. Both quaternion (A) and Euler angles (B) representations yield similar results, with average errors 27 cm and 6.3° exceeding our approach. The superior performance of the keypoint-based and 6D representations, compared to the discontinuous quaternion and Euler angle representations, likely stems from their continuous

nature, as discussed in [33]. Tuning rewards for pose tracking with separate terms for position and orientation proved challenging, due to the difficulty of balancing two quantities with different units and magnitudes. Achieving both position and orientation tracking in (A), (B), and (C) required many iterations, and often, the training would collapse, prioritizing either position or orientation tracking but rarely both. In contrast, the keypoint-based pose representation required far less tuning due to its unified representation of both aspects.

Table I presents the average position and orientation errors on flat terrain for different added masses on the end-effector in simulation. Within the training range  $[0,2.0]~\mathrm{kg}$ , the tracking errors remain stable, with an average position error of  $0.83~\mathrm{cm}$  and orientation error of  $3.45^\circ$ . When the added mass exceeds the training distribution, the tracking performance degrades with errors reaching  $15.33~\mathrm{cm}$  and  $45.02^\circ$  for a  $4.5~\mathrm{kg}$  load. This highlights the controller's robustness beyond the training range while also demonstrating its limitations when encountering significantly higher payloads.

2) Comparison to model-based control: We compare the tracking performance of our whole-body RL policy with the model-based whole-body MPC controller from prior work [19] optimized for the same robot. For this comparison, we fine-tuned the original weight parameters to optimize wholebody behavior. To enable more natural movement, we set the MPC base reference weights to zero. We evaluate both controllers on flat terrain using the same 35 end-effector pose commands in the expanded workspace (Figure 3B). Both controllers perform similarly in terms of median errors, with the RL controller achieving 1.81 cm / 1.73°, and the MPC controller  $2.17~\mathrm{cm}$  /  $1.53^{\circ}$ . However, the mean errors are notably higher for the MPC controller, reaching 6.43 cm / 6.88°, while the RL controller maintains significantly lower values at 2.21 cm / 2.01°. This discrepancy results from the MPC controller's inability to manage the trade-off between pose tracking and self-collision avoidance, causing the arm to get stuck near the base during transitions between distant poses – an issue that our RL controller effectively avoids.

Since the MPC controller was tuned specifically for flat terrain, we did not compare its performance with our whole-body RL policy on stairs. Unlike our RL policy, the MPC controller would require a terrain model to handle rough terrain, which was beyond the scope of our evaluation.

#### B. Hardware experiments

- 1) Pose tracking accuracy: We assess our controller's tracking performance using a motion capture system across 20 randomly sampled poses in the expanded workspace (Figure 3B). Poses are sent sequentially, with substantial changes in position and orientation, resulting in effective whole-body behavior as shown in Figure 1. The average error reaches 2.03 cm and 2.86°. These results, which are illustrated in Figure 5, closely match the performance observed in simulation, demonstrating a minimal sim-to-real gap.
- 2) Robustness to external disturbances: We evaluate the tracking performance of our whole-body RL policy on stairs using a motion capture system across 20 sampled poses in the

<span id="page-5-0"></span>TABLE I: Average position  $\bar{e}_p$  and orientation  $\bar{e}_o$  errors for different added masses on the end-effector  $m_a$  for 10000 end-effector pose commands on flat terrain in simulation.

| $m_a$ [kg]                         | [0 - 2.0]    | 2.5          | 3.0 | 3.5           | 4.0 | 4.5 |
|------------------------------------|--------------|--------------|-----|---------------|-----|-----|
| $\bar{e}_p$ [cm] $\bar{e}_o$ [deg] | 0.83<br>3.45 | 1.18<br>6.99 |     | 4.77<br>22.54 |     |     |

<span id="page-5-1"></span>![](_page_5_Figure_9.jpeg)

Fig. 5: Distribution of the position and orientation errors for 20 end-effector pose commands, measured on hardware, on both flat terrain and stairs.

expanded workspace (Figure 3B) in the half-space in front of the robot under three base orientations: sideways on the stairs, facing up and facing down, as shown in Figure 1. The average position error reaches  $2.64~\rm cm$ , and the average orientation error  $3.64^{\circ}$ . Figure 5 shows that the tracking performance remains consistent with that on flat terrain.

When switching orientations on the stairs a locomotion policy is used [36] and the transition between policies is smooth thanks to the robot initialization process described in Section III-G. Without this initialization step, the robot experiences jumps when transitioning between policies.

Additionally, the system can handle up to 3.75 kg of weight on the end-effector when stationary, and up to 1.3 kg in movement. This flexibility is advantageous as it avoids the need to model weight changes, typically required in model-based approaches, allowing for the attachment of various end-effectors and dynamic carrying of unknown payloads during operation.

### V. CONCLUSION

We have presented a whole-body RL-based controller for a quadruped with an arm that can reach even the most difficult poses. Our controller achieves high accuracy also on rough terrain (e.g., on stairs), which we show in real-world experiments with ANYmal with an arm. Additionally, our contributions include providing a formulation for learning pose tracking that is superior to existing methods with poor accuracy or only considering position tracking. Our hardware experiments show an average tracking accuracy of 2.64 cm for position and 3.64° for orientation on challenging terrain. Future work includes integrating 3D environment representations for self-learned collision avoidance, as in [37], and improving tracking performance under heavy, unmodeled payloads by incorporating a memory network (e.g., LSTM [38]) or a concurrent state estimation architecture [39].

## REFERENCES

- [1] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, "Learning quadrupedal locomotion on deformable terrain," *Science Robotics*, vol. 5, no. 47, p. eabc5986, 2020.
- [2] T. Miki, J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, "Learning robust perceptive locomotion for quadrupedal robots in the wild," *Science Robotics*, vol. 7, no. 62, 2022.
- [3] G. B. Margolis and P. Agrawal, "Walk these ways: Tuning robot control for generalization with multiplicity of behavior," in *Conference on Robot Learning (CoRL)*, 2023, pp. 22–31.
- [4] C. D. Bellicoso, K. Kramer, M. St ¨ auble, D. Sako, F. Jenelten, ¨ M. Bjelonic, and M. Hutter, "ALMA - Articulated Locomotion and Manipulation for a Torque-Controllable Robot," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2019, pp. 8477–8483.
- [5] J.-P. Sleiman, F. Farshidian, M. V. Minniti, and M. Hutter, "A Unified MPC Framework for Whole-Body Dynamic Locomotion and Manipulation," *IEEE Robotics and Automation Letters (RA-L)*, vol. 6, no. 3, pp. 4688–4695, 2021.
- [6] H. Dai, A. Valenzuela, and R. Tedrake, "Whole-body motion planning with centroidal dynamics and full kinematics," in *IEEE-RAS International Conference on Humanoid Robots*, 2014, pp. 295–302.
- [7] Y. Abe, B. Stephens, M. Murphy, and A. Rizzi, "Dynamic whole-body robotic manipulation," in *Unmanned Systems Technology XV*, vol. 8741, 2013, pp. 280–290.
- [8] J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter, "Learning agile and dynamic motor skills for legged robots," *Science Robotics*, vol. 4, no. 26, p. eaau5872, 2019.
- [9] Z. Fu, X. Cheng, and D. Pathak, "Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion," in *Conference on Robot Learning (CoRL)*, 2023, pp. 138–149.
- [10] X. B. Peng, P. Abbeel, S. Levine, and M. van de Panne, "DeepMimic: Example-guided Deep Reinforcement Learning of Physics-based Character Skills," *ACM Transactions On Graphics*, vol. 37, no. 4, pp. 1–14, 2018.
- [11] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa *et al.*, "Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning," *arXiv preprint arXiv:2108.10470*, 2021.
- [12] P. Arm, M. Mittal, H. Kolvenbach, and M. Hutter, "Pedipulate: Enabling Manipulation Skills using a Quadruped Robot's Leg," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2024, pp. 5717–5723.
- [13] X. Cheng, A. Kumar, and D. Pathak, "Legs as Manipulator: Pushing Quadrupedal Agility Beyond Locomotion," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2023, pp. 5106–5112.
- [14] T. Portela, G. B. Margolis, Y. Ji, and P. Agrawal, "Learning Force Control for Legged Manipulation," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2024, pp. 15 366–15 372.
- [15] B. U. Rehman, M. Focchi, J. Lee, H. Dallali, D. G. Caldwell, and C. Semini, "Towards a multi-legged mobile manipulator," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2016, pp. 3618–3624.
- [16] L. Sentis and O. Khatib, "Synthesis of whole-body behaviors through hierarchical control of behavioral primitives," *International Journal of Humanoid Robotics*, vol. 2, no. 04, pp. 505–518, 2005.
- [17] M. Mittal, D. Hoeller, F. Farshidian, M. Hutter, and A. Garg, "Articulated Object Interaction in Unknown Scenes with Whole-Body Mobile Manipulation," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2022, pp. 1647–1654.
- [18] S. Zimmermann, R. Poranne, and S. Coros, "Go Fetch! Dynamic Grasps using Boston Dynamics Spot with External Robotic Arm," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2021, pp. 4488–4494.
- [19] J.-R. Chiu, J.-P. Sleiman, M. Mittal, F. Farshidian, and M. Hutter, "A Collision-Free MPC for Whole-Body Dynamic Locomotion and Manipulation," pp. 4686–4693, 2022.

- [20] H. Ferrolho, V. Ivan, W. Merkt, I. Havoutis, and S. Vijayakumar, "RoLoMa: Robust loco-manipulation for quadruped robots with arms," *Autonomous Robots*, vol. 47, no. 8, pp. 1463–1481, 2023.
- [21] H. Ferrolho, W. Merkt, V. Ivan, W. Wolfslag, and S. Vijayakumar, "Optimizing Dynamic Trajectories for Robustness to Disturbances Using Polytopic Projections," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2020, pp. 7477–7484.
- [22] S. Jeon, M. Jung, S. Choi, B. Kim, and J. Hwangbo, "Learning Whole-body Manipulation for Quadrupedal Robot," *IEEE Robotics and Automation Letters (RA-L)*, vol. 9, no. 1, pp. 699–706, 2023.
- [23] Y. Ma, F. Farshidian, T. Miki, J. Lee, and M. Hutter, "Combining Learning-Based Locomotion Policy With Model-Based Manipulation for Legged Mobile Manipulators," *IEEE Robotics and Automation Letters (RA-L)*, vol. 7, no. 2, pp. 2377–2384, 2022.
- [24] G. Pan, Q. Ben, Z. Yuan, G. Jiang, Y. Ji, S. Li, J. Pang, H. Liu, and H. Xu, "RoboDuet: Whole-body Legged Loco-Manipulation with Cross-Embodiment Deployment," 2024.
- [25] M. Liu, Z. Chen, X. Cheng, Y. Ji, R. Yang, and X. Wang, "Visual Whole-Body Control for Legged Loco-Manipulation," in *Conference on Robot Learning (CoRL)*, 2024.
- [26] H. Ha, Y. Gao, Z. Fu, J. Tan, and S. Song, "UMI on Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers," in *Conference on Robot Learning (CoRL)*, 2024.
- [27] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, "On the Continuity of Rotation Representations in Neural Networks," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 5745– 5753.
- [28] M. Mittal, C. Yu, Q. Yu, J. Liu, N. Rudin, D. Hoeller, J. L. Yuan, R. Singh, Y. Guo, H. Mazhar, A. Mandlekar, B. Babich, G. State, M. Hutter, and A. Garg, "Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments," *IEEE Robotics and Automation Letters (RA-L)*, vol. 8, no. 6, p. 3740–3747, 2023.
- [29] ANYbotics, "Anymal specifications," 2023, Accessed 1- September-2024. [Online]. Available: [https://www.anybotics.com/](https://www.anybotics.com/anymal-autonomous-legged-robot/) [anymal-autonomous-legged-robot/](https://www.anybotics.com/anymal-autonomous-legged-robot/)
- [30] Duatic, "Dynaarm specifications," 2023, Accessed 1-September-2024. [Online]. Available:<https://duatic.com/>
- [31] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," *arXiv preprint arXiv:1707.06347*, 2017.
- [32] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning," pp. 91–100, 2022.
- [33] A. R. Geist, J. Frey, M. Zhobro, A. Levina, and G. Martius, "Learning with 3D rotations, a hitchhiker's guide to SO(3)," in *Proceedings of the 41st International Conference on Machine Learning*, vol. 235, 2024, pp. 15 331–15 350.
- [34] A. Allshire, M. MittaI, V. Lodaya, V. Makoviychuk, D. Makoviichuk, F. Widmaier, M. Wuthrich, S. Bauer, A. Handa, and A. Garg, ¨ "Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2022.
- [35] Z. Xie, H. Y. Ling, N. H. Kim, and M. van de Panne, "ALLSTEPS: Curriculum-driven Learning of Stepping Stone Skills," in *Computer Graphics Forum*, vol. 39, no. 8, 2020, pp. 213–224.
- [36] P. Arm, G. Waibel, J. Preisig, T. Tuna, R. Zhou, V. Bickel, G. Ligeza, T. Miki, F. Kehl, H. Kolvenbach, and M. Hutter, "Scientific exploration of challenging planetary analog environments with a team of legged robots," *Science Robotics*, vol. 8, no. 80, p. eade9548, 2023.
- [37] T. Miki, J. Lee, L. Wellhausen, and M. Hutter, "Learning to walk in confined spaces using 3D representation," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2024, pp. 8649–8656.
- [38] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
- [39] G. Ji, J. Mun, H. Kim, and J. Hwangbo, "Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion," *IEEE Robotics and Automation Letters (RA-L)*, vol. 7, no. 2, p. 4630–4637, Apr. 2022.

---

## Notes

- **Title:** Whole-Body End-Effector Pose Tracking
- **URL:** https://arxiv.org/pdf/2409.16048
