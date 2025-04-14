import torch
import torch.nn.functional as F
from reconstruction import ConvDenoisingAE
from model import NeuroSeizureNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MemoryBank:
    def __init__(self,  capacity=256, num_negatives=128):
        self.data = torch.tensor([], dtype=torch.float32, device=device)
        self.labels = torch.tensor([], dtype=torch.long, device=device)
        self.ae = ConvDenoisingAE(22,1280).to(device, dtype=torch.float)
        self.causal_model = NeuroSeizureNet().to(device)
        self.ae_optimizer = torch.optim.Adam(self.ae.parameters(), lr=1e-3)
        self.conf_threshold = 0.5
        self.capacity = capacity
        self.num_negatives = num_negatives

    def compute_ite(self, data, labels):
        """计算个体干预效应"""
        with torch.no_grad():
            # 生成反事实特征
            cf_data = self.generate_counterfactuals(data)
            cf_data = cf_data.to(device, dtype=torch.float)
            # 获取预测结果差异
            pred_cf = self.causal_model(cf_data)
            # ITE计算（基于预测差异）
            ite = torch.abs(labels - pred_cf.softmax(dim=1)).sum(dim=1)
        return ite

    def generate_counterfactuals(self, data):
        """生成最小干预反事实样本"""
        noise = torch.randn_like(data) * 0.1
        cf_data = data + noise
        return cf_data

    def update(self, new_data, logits):
        """存储原始数据"""
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            # 计算因果效应得分
            ite_scores = self.compute_ite(new_data,probs)

            # 因果阈值选择（前30%样本）
            sorted_idx = torch.argsort(ite_scores, descending=True)
            topk = int(len(ite_scores) * 0.3)
            topk_mask = torch.zeros_like(ite_scores, dtype=torch.bool)
            topk_mask[sorted_idx[:topk]] = True
            conf_mask = (probs.max(dim=1).values > 0.5)  # 置信度阈值掩码
            combined_mask = topk_mask & conf_mask   # 联合条件

            # 获取符合两个条件的样本索引
            selected_idx = torch.where(combined_mask)

            valid_data = new_data[selected_idx]
            valid_labels = preds[selected_idx]

        self.data = torch.cat([self.data, valid_data])[-self.capacity:].to(new_data.device)
        self.labels = torch.cat([self.labels, valid_labels])[-self.capacity:]
        if len(self.data) > 16:
            self.train_ae()

    def train_ae(self):
        dataset = torch.utils.data.TensorDataset(self.data, self.labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        self.ae.train()
        for _ in range(3):
            for batch_data, _ in loader:
                batch_data = batch_data.to(self.data.device, dtype=torch.float32)
                self.ae_optimizer.zero_grad()
                # 重建
                recon, proj = self.ae(batch_data)
                # 重建损失
                recon_loss = F.mse_loss(recon, batch_data)
                # 相似性保持损失
                sim_loss = -F.cosine_similarity(proj[1:], proj[:-1]).mean()
                # 组合损失
                total_loss = recon_loss + 0.5 * sim_loss
                total_loss.backward()
                self.ae_optimizer.step()

    def sample_negatives(self, anchor_label, device):
        """返回需要重新计算特征的原始数据"""
        if len(self.data) == 0:
            return torch.empty((0, *self.data.shape[1:]), device=device)

        neg_mask = (self.labels != anchor_label)
        if neg_mask.sum() == 0:
            return torch.empty((0, *self.data.shape[1:]), device=device)

        neg_data = torch.as_tensor(self.data[neg_mask]).to(device)
        self.ae.eval()
        with torch.no_grad():
            recon_samples, _ = self.ae(neg_data)

        return recon_samples[torch.randperm(len(recon_samples))[:self.num_negatives]]
