# ESP32-S3-N16R8 Bring-up Notes

当前板子：`ESP32-S3-N16R8`

## 结论
- 这块板子适合做 AirPuff 下位机飞控
- 当前树莓派已经识别到它的串口：`/dev/ttyACM0`
- 现阶段先走 USB 串口链路，不需要先接针脚
- 当前已确认芯片信息：`ESP32-S3`、`16MB Flash`、`8MB Embedded PSRAM`
- 因为 Arduino 工具链下载过慢，建议先用 `MicroPython` 路径完成协议联调

## Arduino IDE 建议配置
- Board: `ESP32S3 Dev Module`
- Port: 板子对应的串口
- USB CDC On Boot: `Disabled` 或保持默认
- Upload Speed: `115200` 或 `460800`
- Flash Size: `16MB`
- PSRAM: `OPI PSRAM`

## 第一阶段目标
1. 能烧录 `esp32_stub.ino`
2. 串口能回 `ACK/STATE/EVENT`
3. 树莓派能通过 `/dev/ttyACM0` 收发命令

## 推荐 bring-up 流程
1. 树莓派上执行 `./pi_flash_esp32_mpy.sh /dev/ttyACM0`
2. 将仓库里的 `esp32_stub_mpy.py` 同步到树莓派，例如放到 `~/esp32_stub_mpy.py`
3. 树莓派上执行 `./pi_upload_esp32_mpy.sh /dev/ttyACM0 ~/esp32_stub_mpy.py`
4. 冒烟测试：`python3 esp32_serial_smoketest.py --port /dev/ttyACM0 --action FORWARD`
5. 系统测试：`python3 airpuff_system_debug.py --server http://192.168.31.240:5000/api/sense --serial /dev/ttyACM0`

## 为什么先走 MicroPython
- 现在的目标是先验证“三端链路”和协议稳定性，不是立刻上电机控制
- MicroPython 能更快把 `ACK/STATE/EVENT` 这套协议跑通
- 后续切回 `esp32_stub.ino` 时，树莓派和笔记本侧仍复用同一协议

## 烧录后快速验证
树莓派上执行：
```
python3 esp32_serial_smoketest.py --port /dev/ttyACM0 --action FORWARD
python3 airpuff_system_debug.py --server http://192.168.31.240:5000/api/sense --serial /dev/ttyACM0
```
