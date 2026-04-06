# ESP32 串口协议（树莓派 -> ESP32）

目标：树莓派只做“上层决策转发”，ESP32 负责电机/PWM/飞控底层。

当前板型：`ESP32-S3-N16R8`
- 可按 `ESP32S3 Dev Module` 目标烧录
- `N16R8` 表示 `16MB Flash + 8MB PSRAM`
- 当前这块板子在树莓派侧已经识别为 `/dev/ttyACM0`
- 当前接上的口已经能当作下载/串口调试口使用
- 如果 Arduino 工具链下载过慢，可先走 `MicroPython + main.py` 快速联调路径

## 物理连接
- USB 直连（推荐）：树莓派通过 `/dev/ttyUSB0` 或 `/dev/ttyACM0` 写串口
- 串口参数：`115200 8N1`

## 报文格式（ASCII 一行）
```
AP,<ACTION>,<ALT>,<TS_MS>\n
```

字段说明：
- `ACTION`：`FORWARD/BACKWARD/LEFT/RIGHT/STOP/UP/DOWN`
- `ALT`：高度目标（整数，当前用 0 保留）
- `TS_MS`：树莓派时间戳（毫秒）

示例：
```
AP,FORWARD,0,1773720607000
AP,LEFT,0,1773720607500
AP,STOP,0,1773720608000
```

## ESP32 回传格式（第一版桥接固件）
```
ACK,<ACTION>,<ALT>,<TS_MS>
STATE,<ACTION>,<ALT>,<AGE_MS>,<ACTIVE|FAILSAFE>
ERR,<TYPE>,...
EVENT,FAILSAFE,STOP,0
```

示例：
```
ACK,FORWARD,0,1773720607000
STATE,FORWARD,0,42,ACTIVE
EVENT,FAILSAFE,STOP,0
```

## 失败保护（建议）
- ESP32 端：若超过 500ms 没收到新指令，自动 `STOP`
- 树莓派端：已具备链路超时 failsafe（`--link-timeout`）

## 树莓派调试脚本
项目里已增加两份调试脚本：
- `python3 esp32_serial_smoketest.py --port /dev/ttyACM0`
- `python3 airpuff_system_debug.py --server http://192.168.31.240:5000/api/sense --serial /dev/ttyACM0`

## 两条固件路径
### 路径 A：Arduino 版桥接固件
- 文件：`esp32_stub.ino`
- 优点：后续迁移到 PWM/电机控制更顺
- 当前阻塞：`arduino-cli` 需要下载较大的 Espressif 工具链

### 路径 B：MicroPython 快速联调固件
- 文件：`esp32_stub_mpy.py`
- 烧录脚本：`./pi_flash_esp32_mpy.sh /dev/ttyACM0`
- 上传脚本：`./pi_upload_esp32_mpy.sh /dev/ttyACM0 ~/esp32_stub_mpy.py`
- 优点：可以先把树莓派 <-> ESP32 协议联通、failsafe、系统调试跑通
- 说明：协议保持与 Arduino stub 一致，后续切回 Arduino 固件时树莓派和笔记本代码无需改动
