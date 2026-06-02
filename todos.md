确认应该是好的的部分：
    ik test
    四条腿的相位

不知道的部分：
    四条腿的轨迹，现在根据画出来的结果来看，四条腿都是上下动而不是前后动的
    但是理论上写的是完全照抄iktest

    我修不好

    在freq=2的情况下狗是能往前蹦着走的，但是stiffness太硬会导致蹦飞出去
    dcconfig下，他是能缓慢的往前跳的

另，两组对角腿的轨迹的幅度有差别，不知道为什么
再，轨迹缺乏一致性，画出来的东西不是每次都完全一致的
据观察应该是HAA摆动带来的问题，但是理论上HAA不应该动，而且在大多数时候，HAA也确实是不动的

今天的 trial and error 记录：
    轨迹高度符号修正（swing 相改为抬腿方向）+ CPG 零位角对齐 ik_test（-40 / -100）
    step_length 的速度命令修正已临时关闭（改为固定基线 step_length，再叠加 residual）
    Mastiff 执行器由 DCMotorCfg 改为 ImplicitActuatorCfg，stiffness=1e6
