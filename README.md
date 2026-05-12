# 无人机低空航拍语义分割 · 2026AIC算法挑战赛


## 赛题概述 · 赛题三: 无人机低空航拍图像语义分割

随着低空经济的蓬勃发展，无人机技术在测绘、安防、农业监测及城市规划等领域的应用日益广泛
无人机低空航拍图像具有视角独特、分辨率高、地物细节丰富等特点，但同时也面临着视角变化大、目标尺度差异显著、背景复杂等挑战
语义分割作为计算机视觉的核心任务之一，旨在对图像中的每个像素进行分类，是实现无人机视觉感知智能化的关键

然而，现有的通用语义分割模型在直接应用于低空航拍场景时，往往因训练数据与航拍数据分布差异（DomainGap）而表现不佳
此外，航拍图像中不同地物类别（如农田、建筑、道路）的分布极不均衡，
且测试场景中某些类别的占比可能与训练场景存在显著偏差（如特定区域农田占比过高），这要求模型具备极强的泛化能力和鲁棒性
本赛题希望通过开发高效的无人机低空航拍图像语义分割模型，探索在复杂分布数据环境下提升模型性能的方法，推动无人机视觉技术的落地应用

---


# 项目交接文档 · v2 补充记录

# 一、v2 核心信息
项目	            内容
模型名	            Swin-S + UPerNet
配置文件	        ~/uav_seg_v2/my_configs/upernet_swin_uav.py
最优权重	        ~/uav_seg_v2/work_dirs/upernet_swin_uav/best_mIoU_iter_40000.pth
最终 mIoU	        74.32（验证集 699 张）
训练步数	        40000 步


# 二、vs v1 技术核心差异
维度	               v1（SegFormer-B2）	                  v2（Swin-S + UPerNet）
Backbone	        MixVisionTransformer-B2         	 Swin Transformer-Small
Head	            SegformerHead（轻量 MLP 多尺度）	  UPerHead（显式 FPN + PPM）
多尺度融合	         隐式，MLP 直接融合四尺度	            显式 FPN 横向融合 + PPM（pool_scales=1,2,3,6）≈ ASPP
Auxiliary Head	    ❌ 无	                            ✅ FCNHead 接 stage2，loss_weight=0.4
损失函数	         CE（class_weight）+ Dice	           CE（class_weight）+ Dice
Barren 权重	         1.0	                               4.0（过高，见缺陷）
Vehicle 权重	     3.0	                               4.0
Background 权重	     1.0	                               1.2
训练步数	         20000	                               40000
crop_size	        512×512	                              384×384（显存限制被迫缩小）
滑窗步长	         341	                               256
域适应	            PhotoMetricDistortion	              PhotoMetricDistortion（RandomGrayscale 被迫删除，见失败记录）
预训练权重来源	     MIT-B2（ImageNet）                    Swin-S（ImageNet-1K）


# 三、逐类结果对比
类别	        v1 IoU	v2 IoU	v1 Acc	v2 Acc	说明
Background	    66.18	66.61	76.34	77.02	微升
Building	    82.27	83.21	92.02	91.61	微升
Road	        76.76	79.46	89.86	89.69	最大进步 +2.70
Water	        86.86	87.65	94.20	94.05	微升
Barren	        45.44	43.74	63.26	84.83	IoU 退步，Acc 暴涨，典型过预测
Vegetation	    85.77	85.26	93.12	92.82	微降
Agricultural	76.10	75.78	87.42	83.38	微降
Vehicle	        73.72	72.82	90.91	93.09	IoU 微降，Acc 反升
mIoU	        74.14	74.32	  —	      —	    +0.18，基本持平


# 四、v2 核心缺陷
缺陷1：Barren 权重 4.0 过激进（最重要）
现象：Barren Acc 63→84（大幅提升），但 IoU 45→43（反而下跌）
机制：
IoU = TP / (TP + FP + FN)
Acc 高 = TP 增加（漏检减少）✅
IoU 低 = FP 同时增加（把其他类误判为 Barren）❌
结论：权重 4.0 导致模型过度预测 Barren，合理范围是 2.0~2.5

缺陷2：crop_size 被迫从 512 降到 384
原因：Swin-S + UPerNet 比 SegFormer-B2 更重，8GB 显存下 512×512 crop 速度不可接受（7秒/iter ≈ 6.6天）
代价：384×384 crop 对大面积目标（Agricultural、Background）的上下文感知变弱
体现：Agricultural IoU 76.10→75.78，Background 几乎没有进步

缺陷3：架构升级收益接近零
Swin-S + UPerNet 在论文上比 SegFormer-B2 更强，但在本赛题 8GB 显存限制下：
crop_size 被迫缩小抵消了架构优势
40000 步仍然不够充分训练一个更重的模型
结论：架构不是本赛题的瓶颈，训练步数和类别权重调整才是

缺陷4：显式域适应未能实现
原计划用 IBN-Net（IN+BN 混合归一化）做显式域适应
失败原因：mmseg 1.2.2 未内置 ResNet_IBN
备用方案 RandomGrayscale 与 bgr_to_rgb=True 冲突被迫删除
最终 v2 的域适应能力与 v1 相同，仅靠 PhotoMetricDistortion 隐式覆盖


# 五、v2 搭建过程中所有失败记录
失败1：ResNet_IBN 不存在
计划：用 ResNet-101-IBN-a 做显式域适应
现象：ImportError: cannot import name 'ResNet_IBN'
原因：mmseg 1.2.2 未内置该 backbone
处理：方案整体废弃，改用 Swin-S

失败2：FocalLoss 不支持 ignore_index
现象：TypeError: __init__() got an unexpected keyword argument 'ignore_index'
原因：mmseg 1.2.2 的 FocalLoss 实现不接受该参数
处理：删除 ignore_index=ignore_index 参数

失败3：FocalLoss 不支持多分类
现象：AssertionError: Only sigmoid focal loss supported now
原因：该版本 FocalLoss 为目标检测二分类设计，use_sigmoid 必须为 True
处理：改为 use_sigmoid=True

失败4：FocalLoss alpha=None 不合法
现象：AssertionError: alpha should be of type float
原因：该版本强制要求 alpha 为 float
处理：改为 alpha=0.25

失败5：FocalLoss 多分类 CUDA 报错
现象：RuntimeError: CUDA error: device-side assert triggered
原因：use_sigmoid=True 的 FocalLoss 按二分类处理 logits，与 9 类标签维度不匹配，触发 CUDA 越界
处理：彻底删除 FocalLoss，不再尝试

失败6：RandomGrayscale 导致单通道崩溃
现象：IndexError: index 1 is out of bounds for dimension 0 with size 1
原因：RandomGrayscale 输出单通道图像，data_preprocessor 的 bgr_to_rgb=True 对 3 通道做 [2,1,0] 重排时越界
处理：删除 RandomGrayscale
修复方向（留给下一个创作者）：可在 RandomGrayscale 后加 dict(type='ToRGB') 强制转回三通道，或关闭 bgr_to_rgb 改在 mean/std 里手动调整顺序

失败7：libcudnn 每次新开终端失效
现象：Could not load library libcudnn_cnn_infer.so.8
原因：LD_LIBRARY_PATH 未持久化，每次新开终端丢失
处理：每次训练前执行:
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH


# 六、留给下一个创作者的明确结论
不要再做的方向	                           原因
继续折腾 FocalLoss	                    mmseg 1.2.2 的实现不支持多分类，无法绕过
换更重的 backbone（Swin-B/L）	        8GB 显存下 crop_size 会被迫进一步缩小，得不偿失
把 Barren/Vehicle 权重设到 4.0 以上	    已验证过预测，IoU 反跌

最值得做的事（优先级排序）
优先级	方向	预期收益	说明
🔴 P0	回到 SegFormer-B2，从 v1 权重继续训到 80000 步	+3~5 mIoU	训练不足是 v1 最大硬伤
🔴 P0	Barren 权重设为 2.5（不是 4.0）	Barren +2~4	已有实验教训
🟠 P1	TTA 推理（水平+垂直翻转取均值）	+0.5~1 mIoU	不需要重新训练
🟠 P1	滑窗步长保持 256	+0.3~0.8 mIoU	v2 已验证有效
🟡 P2	RandomGrayscale 修复后重新加入	域适应收益未知	需解决三通道问题

训练命令（接手后直接可用）Bash
conda activate uav_seg
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd ~/uav_seg
PYTHONPATH=$PWD python ~/mmsegmentation/tools/train.py \
    my_configs/segformer_b2_uav.py \
    --work-dir work_dirs/segformer_b2_uav