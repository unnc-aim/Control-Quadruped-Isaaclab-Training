# debug log

此处记录一些注意事项以及随笔

by Calciiite 2026/04

迷思之一：在刚开始训练的时候，illegal contact是应该给出penalty还是termination？

如果只给penalty，agent最后很可能收敛到一个很诡异的步态，在illegal contanct的情况下进行移动（区一般蠕动）
如果给termination，mean-ep-length很可能会持续的保持在很少的step，在这种情况下即使env=4096，获得的数据也会很有限，可能会导致收敛缓慢

如果先only penalty再进行后序的加入termination的训练呢？

考虑PPO的on policy性，其实agent会形成很强的路径依赖，不给termination会卡在局部最优。后续再给出termination会导致loss爆炸

另，RL的学习逻辑大概可以总结为：在最少动作下（成本最低？）换取最多reward的策略，这会导致像`alive = RewTerm(func=mdp.is_alive, weight=1)`这样的白给奖励将agent引到一个蹲着不动的策略上。