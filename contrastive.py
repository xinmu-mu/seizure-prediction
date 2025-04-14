import torch
import torch.nn.functional as F

def info_nce_loss(anchor, positive, negatives, temperature=0.07):
    device = anchor.device
    batch_size, seq_len, feat_dim = anchor.shape
    # 将时空特征展平为二维 (batch, seq_len*feat_dim)
    anchor_flat = anchor.contiguous().view(batch_size, -1)
    positive_flat = positive.contiguous().view(batch_size, -1)
    # 处理空负样本情况
    if negatives is None or len(negatives) == 0:
        dummy_logit = torch.sum(anchor_flat * 0.0)
        return dummy_logit * 0.0
    # 处理负样本 (n, seq_len, feat_dim) -> (n, seq_len*feat_dim)
    neg_flat = negatives.view(-1, seq_len * feat_dim).to(device)
    # 相似度计算（特征空间）
    pos_sim = F.cosine_similarity(anchor_flat, positive_flat, dim=1)  # (1,)
    neg_sim = F.cosine_similarity(anchor_flat.unsqueeze(1),neg_flat.unsqueeze(0),dim=2)  # (1, n)

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (1, 1+n)
    # 温度缩放
    logits /= temperature
    # 损失计算
    labels = torch.zeros(batch_size, dtype=torch.long).to(device)
    return F.cross_entropy(logits, labels)