import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from data.dataset import MyDataset
from models.convnext import ConvNext
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def evaluate_metrics(model_path, test_json, device='cuda'):
    device = torch.device(device)
    
    # 加载数据
    dataset = MyDataset(test_json, image_size=224)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 加载模型
    model = ConvNext(pretrained=False, num_classes=1)

    checkpoint = torch.load(model_path, map_location=device)
    # 判断，取其中的 'model' 部分
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
      print("取model部分...")
      state_dict = checkpoint['model']
    else:
      state_dict = checkpoint

    # 处理 DDP 产生的 'module.' 前缀 
    new_state_dict = {}

    for k, v in state_dict.items():
        # 去掉 DDP 产生的 'module.' 前缀
        name = k.replace('module.', '')
        
        # 关键：解决 timm 模型内部 head 和自定义 head 的冲突
        # ForensicHub 的 ConvNextSmall 结构：
        # - self.model (timm 基础网络)
        # - self.head (自定义 Linear(768, 1))
        
        # 如果权重来自 ForensicHub 的 self.head，映射到 MCC 模型的 self.head
        # 如果权重来自 ForensicHub 的 self.model，映射到 MCC 模型的 self.model
        
        if name.startswith('head.'):
            new_state_dict[name] = v
        elif name.startswith('model.'):
            if 'head.fc' in name or 'classifier' in name:
                continue
            new_state_dict[name] = v
        else:
            new_state_dict[name] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loading Result: {msg}")

    model.to(device).eval()
    
    all_preds = []
    all_labels = []

    print("Start test...")
    for batch in tqdm(loader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        output = model(images)
        preds = output['pred'] 
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 3. 计算工业级指标
    y_true = np.array(all_labels)
    y_scores = np.array(all_preds)
    y_pred_binary = (y_scores > 0.5).astype(int) # 二值化

    acc = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_scores)

    print("\n" + "="*30)
    print(f"Result:")
    print(f"Accuracy :  {acc:.4f}")
    print(f"Precision : {precision:.4f} ")
    print(f"Recall :    {recall:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print(f"AUC:        {auc:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_metrics(
        model_path="/mnt/data0/liuxin/ForensicHub-master/ForensicHub/log/3_27/ConvNextSmall_train_with_sample/checkpoint-19.pth",
        test_json="/mnt/data0/liuxin/Dataset/datasets_json/Fake_Hongkong/sample.json" 
    )
    #test