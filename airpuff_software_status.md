# AirPuff 软件状态记录

更新时间：`2026-03-20`

## 当前架构
- 笔记本：高层决策、Web 控制面板、后续视觉与自然语言能力
- 树莓派：飞行端上层代理，负责上传数据、接收高层命令、转发到 ESP32
- ESP32-S3-N16R8：下位机执行层，当前已接入串口桥接 stub

## 今日已完成
- ESP32 已通过树莓派成功烧录 `MicroPython`
- `main.py` 已上传为 [esp32_stub_mpy.py](/Users/walter/Desktop/SETM下/机器人/esp32_stub_mpy.py)
- `Pi -> Laptop -> Pi -> ESP32` 三端链路已跑通
- 新增一键启动脚本：
  - [laptop_start_airpuff_server.sh](/Users/walter/Desktop/SETM下/机器人/laptop_start_airpuff_server.sh)
  - [pi_airpuff_stack_check.sh](/Users/walter/Desktop/SETM下/机器人/pi_airpuff_stack_check.sh)
- 新增端到端压测脚本：
  - [airpuff_soak.py](/Users/walter/Desktop/SETM下/机器人/airpuff_soak.py)
- `airpuff_client.py` 已支持：
  - `--max-loops`
  - `--log-path`
  - `--request-timeout`

## 2026-03-20 追加进展
- 笔记本完整 Python 依赖已装齐：
  - `numpy`
  - `opencv-python-headless`
  - `faster-whisper`
- 环境检查脚本 [laptop_check_airpuff_env.py](/Users/walter/Desktop/SETM下/机器人/laptop_check_airpuff_env.py) 已确认：
  - `minimal_ready = true`
  - `vision_lite_ready = true`
  - `vision_flow_ready = true`
  - `full_ready = true`
- 笔记本 `vision-lite` 模式已实际启动并通过健康检查
- 新的 ESP32 执行层脚本 [esp32_exec_mpy.py](/Users/walter/Desktop/SETM下/机器人/esp32_exec_mpy.py) 已上板运行
- 新执行层已实际回传：
  - `ACK`
  - `STATE`
  - `STATECTL`
  - `EVENT,FAILSAFE,STOP,0`
- `full` 启动 profile 已调整为更贴近飞艇实际飞行的默认配置：
  - `AIRPUFF_VISION_MODE=flow`
  - `AIRPUFF_ENABLE_VLM=0`
  - `AIRPUFF_CMD_FAST_PATH=1`
- `airpuff_server.py` 已新增：
  - 文本控制 `cmd_fast` 快路径
  - 聊天 `chat_direct` 直达路径
  - `route` 回传与日志落盘
  - 修复“聊天请求耗时过长时被 autopilot 抢占”的危险行为

## 已验证结果
### 快速联调
- `airpuff_system_debug.py` 正常返回笔记本健康状态
- `esp32_serial_smoketest.py` 正常返回：
  - `ACK`
  - `STATE`
  - `EVENT,FAILSAFE,STOP,0`

### 80 轮 soak test 结果
- Brain HTTP 平均延迟：`88.81 ms`
- Brain HTTP P95：`106.01 ms`
- Brain HTTP 最小/最大：`21.27 / 126.09 ms`
- ESP32 ACK 平均延迟：`19.89 ms`
- ESP32 ACK P95：`24 ms`
- Failsafe 平均触发时间：`524.84 ms`
- Failsafe P95：`529 ms`
- 总失败数：`0`

结论：
- 当前最小链路在文本控制模式下已经稳定
- 串口与 failsafe 时序正常
- 对“远端大脑 + 近端转发 + 下位机执行”的软件结构是可行的

### 2026-03-20 非 VLM 全量软件压测
- 笔记本本机服务压测结果：
  - `sense_cmd_fast`：`avg 2.29 ms`，`p95 2.82 ms`
  - `sense_chat_direct`：`avg 3527.10 ms`，`p95 4729.17 ms`
  - `sense_flow`：混合节拍下 `avg 6.87 ms`
  - 按 `flow` 节拍（250ms）单独测试时：
    - `vision_flow_count = 12 / 12`
    - `avg 13.97 ms`
    - `max 14.96 ms`
- Whisper 本机基准：
  - `avg 460.16 ms`
  - `p95 539.60 ms`
- 树莓派 `Pi -> Laptop -> ESP32` 端到端 60 轮 soak：
  - `failures = 0`
  - Brain HTTP：`avg 93.28 ms`，`p95 106.58 ms`
  - ESP32 ACK：`avg 17.42 ms`，`p95 21 ms`
  - Failsafe：`avg 520.80 ms`，`p95 525 ms`
  - 全部命中 `route=cmd_fast`

对应日志：
- 笔记本运行时基准：`~/airpuff_runtime_bench_20260320_011914.json`
- 树莓派 soak summary：`~/airpuff_logs/full_soak_20260320_011819_summary.json`
- 树莓派 soak 明细：`~/airpuff_logs/full_soak_20260320_011819.jsonl`

## 当前运行中的内容
- 笔记本 `full`（`flow + cmd_fast + no VLM`）模式服务正在运行
- 笔记本事件日志落盘到：`~/airpuff_server_events.jsonl`
- 树莓派长时间 soak 测试日志落盘到：`~/airpuff_logs/`

## 主要风险
- 树莓派出现过 `Undervoltage detected!`
- 虽然笔记本完整依赖已装好，但真实摄像头上还需要继续验证 `flow/lite` 在强光、反光、低纹理场景下的表现
- ESP32 当前已是“执行层骨架”，但仍未接入真实 PWM/电机/传感器闭环

## 下一阶段建议
1. 给 ESP32 增加真实 PWM/舵机/电机映射与最小本地安全状态机
2. 用真实 1080p 摄像头做 `flow/lite` 强光与低纹理场景测试
3. 将树莓派客户端接成稳定的常驻进程模式
4. 在供电稳定后做更长时间的联机稳定性测试
