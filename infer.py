import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.convnext import ConvNext

class SimpleFolderDataset(Dataset):
    """
    一个简单的 Dataset，用于读取文件夹内所有图片
    """
    def __init__(self, folder_path, image_size=224):
        self.folder_path = folder_path
        self.image_names = [f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, img_name

def load_forensichub_checkpoint(model, checkpoint_path):
    """
    加载并转换权重
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # 获取权重字典，处理可能存在的包装层
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 DDP 产生的 'module.' 前缀
        name = k.replace('module.', '')
        
        # 过滤掉不匹配的层（timm 自带的 1000 类 head）
        if 'head.fc' in name or 'classifier' in name:
            continue
        new_state_dict[name] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"权重加载状态: {msg}")
    return model

@torch.no_grad()
def batch_inference(model_path, data_dir, batch_size=32, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    #test
    # 初始化模型
    model = ConvNext(num_classes=1, pretrained=False)
    model = load_forensichub_checkpoint(model, model_path)
    model.to(device).eval()

    # 数据加载器
    dataset = SimpleFolderDataset(data_dir, image_size=224)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    results = []
    print(f"Start infer,total {len(dataset)} ")
    
    for images, names in tqdm(dataloader):
        images = images.to(device)
        
        output = model(images)
        # 获取 Sigmoid 后的得分
        preds = output['pred'].cpu().numpy() 
        
        for name, score in zip(names, preds):
            results.append({
                'file_name': name,
                'score': float(score),
                'label': 'Fake' if score > 0.5 else 'Real'
            })

    print("\n" + "="*50)
    print(f"{'Name':<30} | {'score':<10} | {'result'}")
    print("-" * 50)
    for res in results:  # 仅展示前20个
        print(f"{res['file_name']:<30} | {res['score']:.4f} | {res['label']}")

    print("="*50)

    return results

if __name__ == "__main__":
    # 配置路径
    CKPT = "/mnt/data0/liuxin/MCC/checkpoints/phase1_baseline/best_baseline.pth"
    IMG_DIR = "/mnt/data0/liuxin/ForensicHub-master/ForensicHub/fakeHongkong_test/Fake"
    
    batch_inference(CKPT, IMG_DIR, batch_size=64)