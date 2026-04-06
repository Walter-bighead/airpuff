# 无路由器演示：笔记本 ↔ 树莓派互联方案

本项目目录里没有现成的离线路由方案说明，这里给出一套稳定可复用的离线互联办法。

## 方案 A（推荐）：笔记本开 Wi‑Fi 热点，树莓派连接

优点：设备少、布线简单、演示更顺滑。  
假设笔记本系统为 Ubuntu（NetworkManager 可用）。

### 1) 笔记本开启热点
1. 找到 Wi‑Fi 网卡名：
   ```
   nmcli dev status
   ```
2. 开热点（示例 SSID/密码）：
   ```
   nmcli dev wifi hotspot ifname <wifi_iface> ssid AirPuff_AP password airpuff1234
   ```
3. 查看热点 IP（一般是 `10.42.0.1`）：
   ```
   nmcli -f GENERAL.CONNECTION,IP4.ADDRESS dev show <wifi_iface>
   ```

### 2) 树莓派连接热点
```
nmcli dev wifi connect AirPuff_AP password airpuff1234
nmcli -f IP4.ADDRESS dev show wlan0
```

### 3) 运行 AirPuff
在树莓派上把 `AIRPUFF_SERVER_URL` 指向笔记本 IP：
```
export AIRPUFF_SERVER_URL=http://<laptop_ip>:5000/api/sense
python3 /home/walter/airpuff_client.py --mode hardware
```

### 4) 联通性检查
树莓派：
```
ping -c 3 <laptop_ip>
curl -s http://<laptop_ip>:5000/api/health
```

## 方案 B：有线直连（以太网/USB 网卡）

### 1) 笔记本设置静态 IP
```
sudo ip addr add 192.168.50.1/24 dev <iface>
```

### 2) 树莓派设置静态 IP
```
sudo ip addr add 192.168.50.2/24 dev eth0
```

### 3) 指向服务地址
```
export AIRPUFF_SERVER_URL=http://192.168.50.1:5000/api/sense
```

## 演示前检查清单
1. 笔记本服务已启动：`/api/health` 返回 ok。  
2. 树莓派能 `ping` 通笔记本。  
3. 树莓派 `AIRPUFF_SERVER_URL` 指向正确 IP。  
4. 现场无干扰 Wi‑Fi 或者手动固定频道。

