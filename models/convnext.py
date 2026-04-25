import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class ConvNext(nn.Module):
  def __init__(self,num_classes=1,pretrained=True,checkpoint_path=None):
    '''
    专门用于分类任务设计的ConvNext模型
    Args:
      pretrained(bool):是否加载预训练权重
      num_classes(int):分类数量,二分类任务设为1,配合Sigmoid输出概率
    '''
    #加载骨干网络
    #使用timm加载convnext_small变体
    super(ConvNext,self).__init__()

    #如果指定了pth路径，就关掉timm的自动下载
    actual_pretrained = pretrained if checkpoint_path is None else False
    self.model = timm.create_model(
      'convnext_small',
      pretrained=pretrained,
      num_classes=num_classes
    )

    #获取特征提取部分
    #timm的forward_features会在全局池化层之前就停止
    self.backbone = self.model.forward_features

    #定义分类头
    #获取特征维度
    out_channels = self.model.num_features

    self.head = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),#全局平均池化
      nn.Flatten(),#展平为向量[B,768]
      nn.Linear(out_channels,num_classes)#映射到最终分类[B,1]
    )

    if checkpoint_path is not None:
          self._load_checkpoint(checkpoint_path)

  def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu',weights_only=False)
        #训练后的权重存在包装
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v


        # --- 核心修改：手动过滤尺寸不匹配的层 ---
        model_dict = self.state_dict() # 获取当前模型的参数结构
        matched_dict = {}
        
        for k, v in new_state_dict.items():
            if k in model_dict:
                # 检查形状是否一致
                if v.shape == model_dict[k].shape:
                    matched_dict[k] = v
                else:
                    # 打印一下哪些层被跳过了，方便调试
                    print(f"⚠️ 跳过层 {k}: 形状不匹配 (预训练 {v.shape} vs 当前 {model_dict[k].shape})")
            else:
                print(f"ℹ️ 跳过层 {k}: 不在当前模型结构中")
            
        # strict=True 要求结构完全一致，微调时建议开启以防加载错版本
        self.load_state_dict(matched_dict, strict=False)
        print(f"✅ 已成功从本地加载微调权重: {path}")

  def forward(self, image, label=None):
    '''
    前向传播
    Args:
        image:输入张量[Batch,3,224,224]
        label:真是标签,用于训练
    '''
    #特征提取
    features = self.backbone(image)

    #得到分类结果
    logits = self.head(features)

    if logits.shape[1] == 1:
      logits = logits.squeeze(dim=1)

    output = {
      "pred":torch.sigmoid(logits),
    }   

    if label is not None:
      loss=F.binary_cross_entropy_with_logits(logits, label.float())
      output["backward_loss"] = loss
    # output = {
    #   "pred":torch.sigmoid(logits),
    #   "backward_loss":loss,
    #   "visual_loss":{
    #     "combined_loss":loss
    #   }
    # }
    #test
    return output