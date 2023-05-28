import torch
from torch import nn

WIFI_VECTOR_SIZE = 95

class CNN(nn.Module):   # output_size为输出类别（2个类别，0和1）,三种KERNEL，size分别是3,4，5，每种KERNEL有100个
    def __init__(self, vocab_size, embedding_dim, x_size, y_size, filter_num=100, KERNEL_lst=(3,4,5), dropout=0.5):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([  # 1表示channel_num，filter_num即输出数据通道数，卷积核大小为(KERNEL, embedding_dim)
                             nn.Sequential(nn.Conv2d(1, filter_num, (KERNEL, embedding_dim)),
                                           nn.BatchNorm2d(filter_num, 0.8),
                                           nn.ReLU(),
                                           nn.MaxPool2d((WIFI_VECTOR_SIZE - KERNEL + 1, 1)))
                            for KERNEL in KERNEL_lst])
        # self.convs1 = nn.Sequential(nn.Conv2d(filter_num, 1))
        # self.convs2 = nn.Sequential(nn.Conv2d(filter_num, 1))
        self.fc_get_x_1 = nn.Sequential(nn.Linear(filter_num * len(KERNEL_lst), int((filter_num * len(KERNEL_lst)+x_size)/2)),
                                        nn.ReLU())
        self.fc_get_x_2 = nn.Sequential(nn.Linear(int((filter_num * len(KERNEL_lst)+x_size)/2), int(((filter_num * len(KERNEL_lst)+x_size)/2+x_size)/2)),
                                        nn.ReLU())
        self.fc_get_x_3 = nn.Linear(int(((filter_num * len(KERNEL_lst)+x_size)/2+x_size)/2), x_size)
        self.fc_get_y_1 = nn.Sequential(nn.Linear(filter_num * len(KERNEL_lst), int((filter_num * len(KERNEL_lst)+y_size)/2)),
                                        nn.ReLU())
        self.fc_get_y_2 = nn.Sequential(nn.Linear(int((filter_num * len(KERNEL_lst)+y_size)/2), int(((filter_num * len(KERNEL_lst)+y_size)/2+y_size)/2)),
                                        nn.ReLU())
        self.fc_get_y_3 = nn.Sequential(nn.Linear(int(((filter_num * len(KERNEL_lst)+y_size)/2+y_size)/2), int((((filter_num * len(KERNEL_lst)+y_size)/2+y_size)/2+y_size)/2)),
                                        nn.ReLU())
        self.fc_get_y_4 = nn.Linear(int((((filter_num * len(KERNEL_lst)+y_size)/2+y_size)/2+y_size)/2), y_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)       # [16, 107, 50](batch, seq_len, embedding_dim)
        x = x.unsqueeze(1)          # [128, 1, 20, 200] 即(batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]

        out = torch.cat(out, dim=1)      # [128, 300, 1, 1]
        out = out.view(x.size(0), -1)    # [128, 300]
        # out = self.dropout(out)
        out_x = self.fc_get_x_1(out)
        out_x = self.fc_get_x_2(out_x)
        logit_x = self.fc_get_x_3(out_x)
        out_y = self.fc_get_y_1(out)
        out_y = self.fc_get_y_2(out_y)
        out_y = self.fc_get_y_3(out_y)
        logit_y = self.fc_get_y_4(out_y)
        # logit_x = self.fc1(out)             # [16, x_size]s
        # logit_y = self.fc2(out)             # [16, y_size]
        return logit_x, logit_y