## 当前配置

使用 `assets/LynxC_fixed.usd` 作为机器人模型
相关关节/零位/坐标系定义在`assets/lynxc_defnitions`

## 结构
`tasks/mdp/ik_lynxc_quadruped_test.py` 数学模型的仿真验证

`tasks/mdp/lynx_gait_generator.py` 实际仿真里用的运动学解算
`tasks/mdp/lynx_gait_action.py` 解算->机器人的关节运动


`scripts/zero_agent.py` 在没agent的情况下看解算是不是对的
`scripts/rsl_rl/train.py` 训练
`scripts/rsl_rl/play.py` 测试

`tasks/lynx_flat_task.py` task环境

## 环境
1. Visual `source ./visual/bin/activate` 用来跑ik相关的验证

2. isaac `source ~/isaac_env.sh` 用来跑模拟器

## 运行
`python ./train.py --task lynx-flat-v0 --num-envs 4096 --headless`
`python ./train.py --task lynx-terrain-v0 --num-envs 4096 --headless`

从checkpoint继续则加上 --resume