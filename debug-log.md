# debug log

此处记录一些注意事项以及随笔

by Calciiite 2026/04
以敬后人

迷思之一：在刚开始训练的时候，illegal contact是应该给出penalty还是termination？

如果只给penalty，agent最后很可能收敛到一个很诡异的步态，在illegal contanct的情况下进行移动（区一般蠕动或者几乎不动）
如果给termination，mean-ep-length很可能会持续的保持在很少的step，在这种情况下即使env=4096，获得的数据也会很有限，可能会导致收敛缓慢

关于很少的step的定义：经验来看，至少如果mean_ep_length只有2-3的话，获得的data是不够agent在~100个iterations里取得任何成长的（表现就是reward和length都不变），不确定scale到3k ep时会不会有差别，或许值得测试一下。

如果先only penalty再进行后序的加入termination的训练呢？

考虑PPO的on policy性，其实agent会形成很强的路径依赖，不给termination会卡在局部最优。后续再给出termination会导致loss爆炸

另，RL的学习逻辑大概可以总结为：在最少动作下（成本最低？）换取最多reward的策略，这会导致像`alive = RewTerm(func=mdp.is_alive, weight=1)`这样的白给奖励将agent引到一个蹲着不动的策略上。或者站着不动，取决于base_height的引导reward会引导到多高。

理论上讲，reward需要和任务目标绑定。

再，base_reward的scale不宜太大，狗的运动一定会导致高度波动，此处给太大惩罚可能会使得agent再走路时采取诡异的策略

--04.16 update--

在采用CPG config的模式下，RL的输出不应该是对CPG行为的控制（因为CPG本身就是一套速度指令->关节动作的映射），具体来说 在0Agent模式下，一个良定义的CPG config应该至少是能让狗（某种程度上稳定的）运动的。

在这种情况下，RL的优化目标就不是给定观测下的行为，而是基于给定观测去调整CPG动作的参数（在这里是步幅，步高，频率，转向比率）。

对大部分的狗构型，CPG_GROUND_HEIGHT_M （髋关节到站立参考平面的距离）可以近似设置为站姿下base_height的相反数，当然最稳妥的办法还是找机械要图纸参考。

但需要注意的是，CPG_GROUND_HEIGHT_M 理论上不适宜作为学习项，在非平地环境，四条腿需要适应不同的离地高度，而这个参数是对整体姿态生效的。

另外，在 CPG 设计下（固定对角相位 + 共享步态参数），步态对称性是结构先验，一般不需要再单独加 gait_diagonal_symmetry 奖励。

再，今天的训练输出验证了之前的假设，即从最开始就给出termination和penalty是必要的，在设计没有问题的情况下，在开始的几十个ep内就能看出收敛的趋向。

> agent在~100个iterations里不取得任何成长（表现就是reward和length都不变）

这个状态表示一些环境设计本身就是有问题的