# 城市低空应急物资投送无人机智能避障仿真

本项目用于《飞行智能控制仿真与场景实践（产业）》期末作品：围绕低空经济中的城市应急医疗物资投送场景，构建二维城市低空空域仿真，演示无人机从医院起飞、绕开建筑/禁飞区/动态空域障碍并到达目标点的过程。

## 快速运行

```powershell
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
```

运行后会生成：

- `outputs/final_trajectory.png`：仿真轨迹截图
- `outputs/drone_delivery_demo.gif`：演示动图
- `outputs/drone_delivery_demo.mp4`：演示视频
- `outputs/summary.json`：实验指标

## 项目结构

```text
.
├─ docs/
│  ├─ 作品文档.md
│  └─ 提示词与调试记录.md
├─ outputs/
│  ├─ final_trajectory.png
│  ├─ drone_delivery_demo.gif
│  ├─ drone_delivery_demo.mp4
│  └─ summary.json
├─ src/
│  └─ drone_delivery_sim.py
├─ requirements.txt
└─ run_demo.ps1
```

## 核心技术

- A* 网格路径规划：在建筑物、禁飞区和安全缓冲区约束下生成全局可行路径。
- 路径简化与前视跟踪：对 A* 节点序列进行视线检测，并用前视目标点减少飞行抖动。
- 简化激光雷达避障：在飞行过程中发射 360 度扫描射线，基于命中点生成局部排斥向量。
- 动态障碍规避：模拟巡检无人机通道和临时吊装设备等低空动态风险。
- 指标验证：输出飞行时间、航程、最小安全距离、是否碰撞等结果。

## 当前实验结果

本地已完成一次仿真运行，结果为：成功到达目标点、无碰撞、飞行时间 35.00 s、实际航程 145.91 m、最小安全距离 1.06 m。
