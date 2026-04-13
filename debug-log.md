# debug log

此处记录一些注意事项以及随笔

by Calciiite 2026/04
以敬后人

迷思之一：在刚开始训练的时候，illegal contact是应该给出penalty还是termination？

如果只给penalty，agent最后很可能收敛到一个很诡异的步态，在illegal contanct的情况下进行移动（区一般蠕动）
如果给termination，mean-ep-length很可能会持续的保持在很少的step，在这种情况下即使env=4096，获得的数据也会很有限，可能会导致收敛缓慢

关于很少的step的定义：经验来看，至少如果mean_ep_length只有2-3的话，获得的data是不够agent在~100个iterations里取得任何成长的（表现就是reward和length都不变），不确定scale到3k ep时会不会有差别。

如果先only penalty再进行后序的加入termination的训练呢？

考虑PPO的on policy性，其实agent会形成很强的路径依赖，不给termination会卡在局部最优。后续再给出termination会导致loss爆炸

另，RL的学习逻辑大概可以总结为：在最少动作下（成本最低？）换取最多reward的策略，这会导致像`alive = RewTerm(func=mdp.is_alive, weight=1)`这样的白给奖励将agent引到一个蹲着不动的策略上。或者站着不动，取决于base_height的引导reward会引导到多高。

理论上讲，reward需要和任务目标绑定。

再，base_reward的scale不宜太大，狗的运动一定会导致高度波动，此处给太大惩罚可能会使得agent再走路时采取诡异的策略

