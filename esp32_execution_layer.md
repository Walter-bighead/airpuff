# ESP32 执行层骨架说明

当前新增文件：`esp32_exec_mpy.py`

目标：在不破坏现有串口协议的前提下，把 ESP32 从“纯串口桥接”推进到“可接真实输出的下位机框架”。

## 现在已经有的层次
- 串口协议层：继续兼容 `AP,<ACTION>,<ALT>,<TS_MS>`
- 状态机：`BOOT / ACTIVE / FAILSAFE`
- 控制目标层：把 `FORWARD / LEFT / UP` 之类的动作映射成 `forward / yaw / vertical`
- 混控层：把三个轴向目标映射成 4 路执行器输出
- 输出层：预留 4 路 PWM 通道，当前默认全部禁用，桌面联调安全

## 混控思路
当前用的是一个对称的 4 通道混控骨架：

```
front_left  = vertical + forward - yaw
front_right = vertical + forward + yaw
rear_left   = vertical - forward - yaw
rear_right  = vertical - forward + yaw
```

这不是最终飞控算法，只是为了先把：
- 命令到执行目标
- 执行目标到输出
- 输出限幅与 failsafe

这三层结构先搭起来。

## 当前默认行为
- 所有输出通道的 `pin=None`
- 即使收到了动作命令，也只会更新内部控制状态，不会真的打 PWM 到外部硬件
- 仍然会正常回：
  - `ACK,...`
  - `STATE,...`
  - `STATECTL,...`
  - `EVENT,FAILSAFE,STOP,0`

## 后面硬件到了怎么接
当你把真实电调或舵机通道定下来以后，只需要改 `esp32_exec_mpy.py` 里的 `OutputBackend()`：
- 给 `front_left/front_right/rear_left/rear_right` 填真实 GPIO
- 必要时调整 `freq / neutral_us / span_us`

## 下一步建议
1. 根据实际推进器布局，确认 4 路输出分别控制什么
2. 若是标准 PWM 电调，先用 50Hz/1000-2000us 做安全占位
3. 接上 MPU6050 / BMP280 后，再在这个结构上补姿态与高度闭环
4. 若最终改走现成飞控，也可以保留这套串口协议层，让 ESP32 只做桥接到飞控的适配
