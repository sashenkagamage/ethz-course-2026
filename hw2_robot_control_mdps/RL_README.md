# Reinforcement Learning vs Inverse Kinematics

This note explains the role of reinforcement learning (RL) in this homework and how it differs from the inverse kinematics (IK) methods used earlier.

## 1. Core Difference: IK vs RL

### Inverse Kinematics

With inverse kinematics, we write explicit mathematical rules to compute joint configurations from a desired end-effector position.

IK answers the question:

> Given this target position, what joint angles should the robot use?

IK is mainly geometric. It focuses on the relationship between joint angles and end-effector position. However, basic IK does not inherently reason about physical effects such as gravity, inertia, motor torque limits, friction, or long-term control behavior.

### Reinforcement Learning

With reinforcement learning, we do not directly tell the robot how to move. Instead, we define what we want the robot to achieve through a reward function.

RL answers the question:

> Given the current state of the robot, what action should the policy take to maximize future reward?

The robot learns through trial and error in simulation. Over time, the policy learns which actions tend to bring the end-effector closer to the target.

## 2. Role of RL in This Exercise

In this homework, the RL agent is a neural network policy trained with PPO.

The policy acts like the robot's controller. It receives an observation of the current robot state and outputs an action. The goal is to control the robot joints so that the end-effector reaches and tracks randomly placed target points.

Instead of solving IK equations at every step, the policy learns a mapping:

```text
observation -> action
```

In this setup:

- The observation describes the robot and target state.
- The action is a normalized joint command.
- MuJoCo simulates how the robot moves.
- The reward tells PPO whether the action was good.

## 3. RL Training Loop

The training loop has four main parts:

```text
Observation -> Policy Action -> MuJoCo Simulation -> Reward -> PPO Update
```

### Step 1: Observation

The environment calls `get_obs()` from `exercises/ex3.py`.

This function collects information such as:

- Current joint positions, `qpos`
- End-effector position in the robot base frame, `ee_pos_base`
- End-effector orientation in the robot base frame, `ee_quat_base`
- Target position in the robot base frame, `target_pos_base`

This observation vector is passed into the policy network.

### Step 2: Action

The policy network outputs an action.

The action is not directly a Cartesian target point. It is a normalized vector with values in `[-1, 1]`, one value per controlled joint.

The function `process_action()` converts this normalized action into actual joint target positions using the robot joint limits:

```text
-1 -> lower joint limit
 0 -> midpoint of joint range
 1 -> upper joint limit
```

### Step 3: MuJoCo Simulation

The processed joint targets are written into `data.ctrl`.

MuJoCo then simulates the robot physics, including effects such as gravity, inertia, joint dynamics, actuator behavior, and contact or friction if present.

The simulator runs at `500 Hz`, with a timestep of `0.002 s`. The environment uses `ctrl_decimation = 50`, meaning the same policy action is held for 50 small physics steps:

```text
0.002 s * 50 = 0.1 s
```

So the policy gives one action every `0.1 s`, or `10 Hz`.

### Step 4: Reward

After the robot moves, the environment computes the end-effector tracking error:

```text
ee_tracking_error = distance between end-effector and target
```

Then it calls `compute_reward()` from `exercises/ex3.py`.

The reward combines:

- A dense reward, which smoothly increases as the end-effector gets closer to the target.
- A sparse reward, which gives an extra bonus when the end-effector is very close to the target.

The current reward is:

```python
dense_reward = np.exp(-2 * ee_tracking_error)
sparse_reward = 1.0 if ee_tracking_error < 0.005 else 0.0
reward = dense_reward + sparse_reward
```

This reward tells PPO whether the recent behavior was useful.

## 4. How PPO Learns

At the start of training, the neural network policy has random weights. Its actions are mostly random, so the robot may move poorly or miss the target.

PPO improves the policy through repeated training cycles:

1. Collect rollouts: Run the current policy in the environment and record observations, actions, and rewards.
2. Evaluate behavior: Identify which actions led to higher rewards and which led to lower rewards.
3. Update the neural network: Adjust the policy weights so that useful actions become more likely in similar states.

Over many iterations, the policy gradually improves. The robot usually transitions from random movement to more stable and accurate reaching behavior.

## 5. Policy vs PPO

It is useful to separate the policy from PPO.

The policy is the neural network. During simulation, it takes the current observation and outputs an action. It does not directly inspect the reward while choosing an action.

PPO is the optimization algorithm. PPO looks at the collected history of observations, actions, and rewards, then updates the policy network so that high-reward actions become more likely in the future.

In short:

```text
Policy = the robot's learned controller
PPO = the training algorithm that improves the policy
```

## 6. Why the Robot May Not Reach the Exact Point

During evaluation, the robot may get very close to the target but not exactly reach it.

This can happen because:

- The action is held for 50 simulation steps.
- The policy outputs joint targets, not exact end-effector positions.
- The reward may be good enough when the robot is close, so the policy may not learn perfect precision.
- The target may be near a difficult region of the workspace.
- The policy is an approximation learned from training, not an exact IK solver.

Getting close is expected. Improving exact tracking usually requires changing the environment design or training setup.

## 7. Possible Improvements

To improve tracking performance, useful changes can be made in `exercises/ex3.py` and, if needed, `env/so100_tracking_env.py`.

### Reward Function

The reward can penalize tracking error more strongly:

```python
dense_reward = np.exp(-10 * ee_tracking_error)
sparse_reward = 2.0 if ee_tracking_error < 0.01 else 0.0
reward = dense_reward + sparse_reward
```

This gives the policy a stronger incentive to reduce the remaining error.

### Observation Function

The observation can include the direct error vector:

```python
target_pos_base - ee_pos_base
```

This tells the policy directly where the target is relative to the end-effector.

### Reset Target Range

Training can be easier if targets are sampled from a smaller reachable region at first:

```python
low=np.array([0.25, -0.15, 0.15])
high=np.array([0.35, 0.15, 0.30])
```

This can help the policy learn stable reaching before trying harder targets.

### Reset Noise

The initial joint noise can be reduced:

```python
np.random.uniform(-0.1, 0.1, size=default_qpos.shape)
```

This makes the starting state less random and can make early learning easier.

### PPO Hyperparameters

The PPO settings in `scripts/train.py` can also affect performance. Examples include:

- `gamma`
- `ent_coef`
- `vf_coef`
- learning rate
- number of environments
- number of training iterations

Changing these can affect exploration, stability, and final tracking performance.

## 8. Summary

IK computes joint configurations using explicit geometry. RL learns a control policy through reward-based trial and error.

In this homework, PPO trains a neural network policy to map observations to joint actions. The policy is evaluated based on how close the end-effector gets to the target. Better rewards, better observations, and more suitable target sampling can improve the learned behavior.