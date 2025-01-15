import torch
import torch.nn as nn
from transformers import BertModel, XLMRobertaModel
from TorchCRF import CRF


class BertMultiTaskModel(nn.Module):
    def __init__(self,config, num_intent, num_slot):
        super(BertMultiTaskModel, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot)
        
        # # 冻结BERT的参数，如果你不希望在微调过程中更新它们
        # for param in self.bert.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None, intent_labels=None, slot_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        
        # 应用 dropout
        sequence_output = self.dropout(sequence_output)
        
        intent_logits = self.intent_classifier(sequence_output[:, 0, :])
        slot_logits = self.slot_classifier(sequence_output[:, :, :])
        
        # 如果提供了标签，则计算损失
        if intent_labels is not None and slot_labels is not None:
            intent_loss_fn = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fn(intent_logits, intent_labels)            
            slot_loss_fn = nn.CrossEntropyLoss()
            slot_mask = slot_labels != -100
            slot_loss = slot_loss_fn(slot_logits.view(-1, slot_logits.size(-1))[slot_mask.view(-1)], slot_labels.view(-1)[slot_mask.view(-1)])
            loss = intent_loss + 2 * slot_loss
            return loss
        else:
            return intent_logits, slot_logits

class BertlstmMultiTaskModel(nn.Module):
    def __init__(self, config, num_intent, num_slot):
        super(BertlstmMultiTaskModel, self).__init__()
        self.bert = XLMRobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent)
        
        # 添加 BiLSTM 层，用于 slot 预测
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot)
        
        # # 冻结BERT的参数，如果你不希望在微调过程中更新它们
        # for param in self.bert.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None, intent_labels=None, slot_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 应用 dropout
        sequence_output = self.dropout(sequence_output)
        
        # 意图分类
        intent_logits = self.intent_classifier(sequence_output[:, 0, :])  # 使用 [CLS] token 的输出
        
        # Slot 分类
        lstm_output, _ = self.bilstm(sequence_output)  # 通过 BiLSTM 层
        slot_logits = self.slot_classifier(lstm_output)  # 计算每个 token 的 slot 分类
        
        # 如果提供了标签，则计算损失
        if intent_labels is not None and slot_labels is not None:
            intent_loss_fn = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fn(intent_logits, intent_labels)
            
            slot_loss_fn = nn.CrossEntropyLoss()
            slot_mask = slot_labels != -100  # 忽略填充部分的标签
            slot_loss = slot_loss_fn(
                slot_logits.view(-1, slot_logits.size(-1))[slot_mask.view(-1)], 
                slot_labels.view(-1)[slot_mask.view(-1)]
            )
            loss = intent_loss + 2 * slot_loss
            return loss
        else:
            return intent_logits, slot_logits  
    
class BertCRFMultiTaskModel(nn.Module):
    def __init__(self, config, num_intent, num_slot):
        super(BertCRFMultiTaskModel, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot)
        self.crf = CRF(num_slot)  # 初始化crf层

    def forward(self, input_ids, attention_mask, token_type_ids=None, intent_labels=None, slot_labels=None):
        # BERT 输出
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        
        # 应用 dropout
        sequence_output = self.dropout(sequence_output)
        
        # Intent 预测
        intent_logits = self.intent_classifier(sequence_output[:, 0, :])  # 取 CLS token 的输出
        
        # Slot 预测
        slot_logits = self.slot_classifier(sequence_output)  # 输出形状 (batch_size, seq_len, num_slot)
        # 训练模式下计算损失
        if intent_labels is not None and slot_labels is not None:
            # Intent 损失
            intent_loss_fn = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fn(intent_logits, intent_labels)
            # Slot 损失（使用 CRF）
            crf_loss = -self.crf.forward(slot_logits, slot_labels, mask=attention_mask.bool())
            
            crf_loss = crf_loss.mean()  # CRF 返回的是一个形状为 (batch_size,) 的张量

            
            # 总损失
            loss = intent_loss + 2 * crf_loss
            return loss
        else:
            return intent_logits, slot_logits
    
    def decode(self, slot_logits, attention_mask):
            # 推理模式，使用 CRF 进行解码
            slot_predictions = self.crf.viterbi_decode(slot_logits, mask=attention_mask.bool())
            
            return slot_predictions
        
class BertlstmCRFMultiTaskModel(nn.Module):
    def __init__(self, config, num_intent, num_slot):
        super(BertlstmCRFMultiTaskModel, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot)

        # 新增 LSTM 层
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.crf = CRF(num_slot)  # 初始化 CRF 层

    def forward(self, input_ids, attention_mask, token_type_ids=None, intent_labels=None, slot_labels=None):
        # BERT 输出
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        
        # 应用 dropout
        sequence_output = self.dropout(sequence_output)
        
        # 通过 LSTM
        lstm_output, _ = self.lstm(sequence_output)  # LSTM 输出形状 (batch_size, seq_len, hidden_size)
        lstm_output = self.dropout(lstm_output)

        # Intent 预测
        intent_logits = self.intent_classifier(lstm_output[:, 0, :])  # 取 CLS token 的输出
        
        # Slot 预测
        slot_logits = self.slot_classifier(lstm_output)  # 输出形状 (batch_size, seq_len, num_slot)

        # 训练模式下计算损失
        if intent_labels is not None and slot_labels is not None:
            # Intent 损失
            intent_loss_fn = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fn(intent_logits, intent_labels)
            
            # Slot 损失（使用 CRF）
            crf_loss = -self.crf.forward(slot_logits, slot_labels, mask=attention_mask.bool())
            crf_loss = crf_loss.mean()  # 如果 CRF 返回的是一个形状为 (batch_size,) 的张量
            
            # 总损失
            loss = intent_loss + 2 * crf_loss
            return loss
        else:
            return intent_logits, slot_logits
    
    def decode(self, slot_logits, attention_mask):
        # 推理模式，使用 CRF 进行解码
        slot_predictions = self.crf.viterbi_decode(slot_logits, mask=attention_mask.bool())
        return slot_predictions
    
    

class IDCNN(nn.Module):
    def __init__(self, input_dim, num_filters):
        super(IDCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,
                      out_channels=num_filters,
                      kernel_size=3,
                      dilation=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_filters,
                      out_channels=num_filters,
                      kernel_size=3,
                      dilation=2,
                      padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_filters,
                      out_channels=num_filters,
                      kernel_size=3,
                      dilation=4,
                      padding=4),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_layers(x)


    
