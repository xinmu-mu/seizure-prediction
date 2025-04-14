import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import data_utils
from data_utils import *
from model import NeuroSeizureNet
from memory_bank import MemoryBank
from contrastive import info_nce_loss
patient_data = data_utils.main_process()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
numPatient = ['chb01', 'chb02', 'chb03', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09','chb10',
              'chb11','chb13','chb14', 'chb16', 'chb17', 'chb18','chb19','chb20','chb21','chb22','chb23']
num_epoch = 50
bS = 64

torch.manual_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


criterion = nn.CrossEntropyLoss()
accuracy_history = []


for patient in  numPatient:
    train_data, train_labels, test_data, test_labels = split_data_by_patient(patient_data, patient)
    trainData, testData, trainLabel, testLabel = (torch.tensor(train_data), torch.tensor(test_data),
                                                  torch.tensor(train_labels), torch.tensor(test_labels))

    train_loader = DataLoader(TensorDataset(trainData, trainLabel), batch_size=bS, shuffle=True)
    test_loader = DataLoader(TensorDataset(testData, testLabel), batch_size=bS, shuffle=False)
    print(f'patient {patient }')
    network = NeuroSeizureNet(num_classes=2).to(device, dtype=torch.float)
    optimizer = torch.optim.Adam(network.parameters(),lr=1e-3)
    memory_bank = MemoryBank(capacity=128)
    epoch_accuracies = []
    patience = 5
    best_accuracy = 0.0
    patience_counter = 0
    for epoch in range(num_epoch):
        # ========== 训练阶段 ==========
        network.train()
        for src_data, src_labels in train_loader:
            # 数据加载
            source_data = src_data.to(device, dtype=torch.float32)
            source_label = src_labels.to(device, dtype=torch.long)
            # 前向计算
            src_pred = network(source_data)
            cls_loss = criterion(src_pred, source_label)
            # 反向传播
            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step()

        # ========== TTA适应阶段 ==========
        optimizer_TTA = torch.optim.SGD(network.classifier.parameters(), lr=1e-4)
        # 评估指标
        testCorrect = 0
        testTotal = 0
        confidence_threshold = 0.5
        for target_data, _ in test_loader:
            # 数据加载
            target_data = target_data.to(device, dtype=torch.float32)
            class_logits = network(target_data)
            class_probs = torch.softmax(class_logits, dim=1)
            confidences = class_probs.max(dim=1).values
            batch_preds = class_logits.argmax(dim=1)

            # 对比学习
            anchor_idx = random.randint(0, target_data.size(0) - 1)
            anchor_label = batch_preds[anchor_idx].item()

            # 计算锚点特征
            anchor = target_data[anchor_idx:anchor_idx + 1]
            anchor_features = network.feature_extractor(anchor)
            # 正样本选择
            pos_mask = ((batch_preds == anchor_label) &
                        (torch.arange(target_data.size(0), device=device) != anchor_idx)
                        &(confidences >= confidence_threshold))
            if pos_mask.sum() == 0:
                continue
            indices = torch.where(pos_mask)
            # 随机选择
            rand_index = random.randint(0, len(indices) - 1)
            pos_idx = indices[rand_index][0]

            # 提取对应数据
            pos_data = target_data[pos_idx:pos_idx + 1]
            positive = network.feature_extractor(pos_data)
            # 负样本采样
            neg_data = memory_bank.sample_negatives(anchor_label,device=anchor.device)
            if neg_data.size(0) == 0:
                neg_feat = torch.zeros_like(positive)
            else:
                neg_feat = network.feature_extractor(neg_data)
            # 对比损失计算
            contrastive_loss = info_nce_loss(anchor_features, positive.unsqueeze(0), neg_feat)
            loss = contrastive_loss
            # 梯度更新
            optimizer_TTA.zero_grad()
            loss.backward()
            optimizer_TTA.step()
            # 更新记忆库
            memory_bank.update(target_data, class_logits)

        # ========== 评估阶段 ==========
        network.eval()
        with torch.no_grad():
            for target_data, target_label in test_loader:
                target_data = target_data.to(device, dtype=torch.float32)
                outputs = network(target_data)
                preds = outputs.argmax(dim=1).cpu()
                testCorrect += (preds == target_label).sum().item()
                testTotal += target_label.size(0)

        testAccuracy = testCorrect / testTotal
        epoch_accuracies.append(testAccuracy)
        average_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)

        #早停逻辑
        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            best_model_weights = network.state_dict().copy()  # 保存最佳模型
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                early_stop_flag = True
                break

        print('Epoch:', epoch, 'Train Loss:',cls_loss.item(), 'Adapt Loss:', loss.item())
    average_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
    accuracy_history.append(average_accuracy)
    print('Test Accuracy:', average_accuracy)


