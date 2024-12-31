虚拟试穿笔记

## Virtual Try-on <a name="Virtual-Try-on"></a>
**虚拟试穿**: 虚拟试穿是通过虚拟的技术手段，不需要真实穿戴，实现变装效果。该技术同时可以扩展到鞋子，眼镜，包等。

**关键挑战**：将服装与各种姿势或手势的人体相匹配，而不会在服装中产生图案和纹理的任何扭曲。

**数据集**：

1）[VITON-HD (High-Resolution VITON-Zalando Dataset)](https://github.com/shadow2496/VITON-HD), 包括上衣和裙子。</br>
2) [dress-code](https://github.com/aimagelab/dress-code), 包括上衣、裤子、裙子3种类别。

以下是对一些论文或项目的知识总结（持续更新中）

- [x] CAT-VTON [前往论文笔记](https://github.com/xuanandsix/awesome-virtual-try-on-note/tree/main/CAT-VTON)
- [x] IDM-VTON [前往论文笔记](https://github.com/xuanandsix/awesome-virtual-try-on-note/tree/main/IDM-VTON)

|方法|试穿模型|辅助模型|文本提示词|可训练参数|
|:--:|:--:|:--:|:--:|:--:|
|CAT-VTON|SD1.5修复模型UNet|无|无|Self Attention|
|IDM-VTON|SDXL修复模型UNet|GarmentNet、IP-Adapter|包括款式、袖长等详细提示词|修复模型UNet、IP-Adapter|
