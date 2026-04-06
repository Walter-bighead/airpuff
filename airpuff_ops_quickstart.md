# AirPuff 软件联调速查

目标：让笔记本、树莓派、ESP32 三端在没有额外解释的情况下也能快速恢复到可测试状态。

硬件装配与接线参考：
- [airpuff_hardware_install_wiring_guide.md](/Users/walter/Desktop/SETM下/机器人/airpuff_hardware_install_wiring_guide.md)

## 角色分工
- 笔记本：运行 `airpuff_server.py`，负责高层决策、界面、后续视觉与自然语言能力
- 树莓派：运行上层客户端和调试脚本，负责链路转发、上传传感器/相机数据、接收笔记本指令
- ESP32-S3：运行低层串口桥接固件，负责未来的 PWM/飞控执行层

## 当前推荐启动方式
### 笔记本
```
chmod +x ~/laptop_start_airpuff_server.sh
~/laptop_start_airpuff_server.sh minimal ~/airpuff_server.py
```

说明：
- `minimal` 会关闭 `LLM/VLM/Whisper`，适合做链路联调和延迟测试
- `vision-lite` 会打开轻量视觉模式
- `vision-flow` 会打开光流视觉模式
- 若后面装好了完整依赖，可改成：
```
~/laptop_start_airpuff_server.sh full ~/airpuff_server.py
```

`full` 当前含义：
- 保留 `LLM + Whisper`
- 默认启用 `cmd_fast` 指令快路径
- 默认使用 `flow` 视觉
- 默认关闭 `VLM`，避免把飞行视觉链路绑到大模型上

仅用于对照实验时，才使用：
```
~/laptop_start_airpuff_server.sh full-vlm ~/airpuff_server.py
```

环境检查：
```
python3 ~/laptop_check_airpuff_env.py
```

安装完整 Python 依赖：
```
chmod +x ~/laptop_install_airpuff_full.sh
~/laptop_install_airpuff_full.sh
```

### 树莓派
快速联调：
```
python3 ~/airpuff_system_debug.py --server http://192.168.31.240:5000/api/sense --serial /dev/ttyACM0
```

快速冒烟：
```
python3 ~/esp32_serial_smoketest.py --port /dev/ttyACM0 --action FORWARD
```

一键堆栈检查：
```
chmod +x ~/pi_airpuff_stack_check.sh
ITERS=30 ~/pi_airpuff_stack_check.sh http://192.168.31.240:5000/api/sense /dev/ttyACM0
```

## ESP32 固件
当前推荐先用 MicroPython 桥接固件做联调。

烧录：
```
~/pi_flash_esp32_mpy.sh /dev/ttyACM0
```

上传主程序：
```
~/pi_upload_esp32_mpy.sh /dev/ttyACM0 ~/esp32_stub_mpy.py
```

下一阶段执行层版本：
```
~/pi_upload_esp32_mpy.sh /dev/ttyACM0 ~/esp32_exec_mpy.py
```

说明：
- `esp32_stub_mpy.py` 是最小桥接版
- `esp32_exec_mpy.py` 是带状态机、混控和执行层骨架的版本

## 日志与输出
- 笔记本服务标准输出：`~/airpuff_server.log`
- 树莓派 soak 测试日志目录：`~/airpuff_logs/`
- `airpuff_client.py` 现已支持 `--log-path` 与 `--max-loops`

## 当前已验证链路
- `Pi -> Laptop /api/sense` 文本命令链路可用
- `Pi -> ESP32` 串口协议 `AP,<ACTION>,<ALT>,<TS_MS>` 可用
- `ESP32 -> Pi` 回传 `ACK/STATE/EVENT` 可用
- `500ms` 左右无指令自动进入 `FAILSAFE`
- `full` 配置下文本指令已命中 `cmd_fast`
- `full` 配置下 `flow` 视觉链路已在本机压测通过

## 当前主要风险
- 树莓派出现过 `Undervoltage detected!`
- 笔记本当前只装了最小依赖，完整视觉/语音能力后续还需要继续补环境
- ESP32 目前还是“桥接 stub”，还没接入真实 PWM/电机输出
