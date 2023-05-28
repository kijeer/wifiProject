<<<<<<< HEAD
from flask import Flask, request, jsonify
import torch
import numpy as np
from CNN import CNN

app = Flask(__name__)

# Load the trained model
model = torch.load('best_CNN_lalala_1.pth')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from the request
    data = request.json['data']

    # Perform inference using the trained model
    input_data = process(data)
    output = model(input_data)
    pos = deal(output)
     # Return the prediction as a JSON response
    return jsonify("position", pos)

def process(data_dict):
    """
    :param data_dict: 在当前位置收集到的 mac address --> ss 字典
    :return: ss特征向量
    """
    mac_list = ['c2:bf:a3:c6:58:96', 'f6:6d:2f:de:45:28', 'f4:84:8d:c2:79:f8', 'f4:6d:2f:64:94:a1', '34:79:16:f3:4d:74', '3c:cd:57:f9:3a:42', '12:82:3d:8e:cc:ea', '0a:f9:f8:02:90:3a', '12:82:3d:8e:cc:ee', 'f4:6d:2f:64:94:9f', '22:0b:20:bf:b6:b9', '16:cb:19:11:97:c9', 'f8:8c:21:e3:e6:a5', 'f8:8c:21:e3:e6:a7', '2a:3a:4d:04:6c:95', 'f2:a6:54:c0:f8:4c', '54:71:dd:00:5a:99', '90:98:38:45:d5:ec', 'fa:6d:2f:64:94:a1', '12:82:3d:5e:cc:da', '12:82:3d:1e:cc:ee', 'f4:84:8d:c2:79:fa', 'c2:69:59:1d:cb:4f', '18:f2:2c:e3:5a:9e', '58:50:ed:f3:f7:82', 'ca:94:02:fc:66:d5', '12:82:3d:8e:cc:da', '12:82:3d:8e:cd:0e', '64:6e:97:d9:c4:b8', 'c2:bf:a3:c6:58:92', '52:84:92:a1:01:08', 'd4:9e:3b:b2:b2:e5', '0a:f9:f8:02:90:38', '44:df:65:6c:ea:d1', '34:79:16:d3:4d:76', '5c:ea:1d:0f:15:0d', 'f4:2a:7d:f1:13:e7', '12:82:3d:de:cc:ee', '12:82:3d:5e:cd:0e', '90:98:38:f5:d5:ed', 'f4:6d:2f:ee:45:28', '12:82:3d:5e:cc:c6', '42:cd:57:f9:3a:42', '12:82:3d:5e:cc:ee', '04:f9:f8:02:90:3a', '12:82:3d:5e:cc:ea', 'f4:6d:2f:ee:45:2a', '54:71:dd:05:8c:35', '22:0b:20:bf:b6:b5', '46:df:65:4c:ea:d0', 'f8:8c:21:15:3e:e3', '12:82:3d:de:cc:ea', 'fe:6d:2f:64:94:9f', '12:82:3d:1e:cc:ea', '68:77:24:27:8e:c4', '12:82:3d:1e:cd:0e', '02:e8:53:d6:0c:c2', '68:77:24:06:e4:4c', '82:6e:f5:3c:f6:be', '44:df:65:6c:ea:d0', '90:98:38:45:d5:fc', '3e:6a:48:28:59:a1', '90:98:38:45:d5:e8', 'c2:bf:a3:c6:58:a1', '90:98:38:45:d5:f8', '68:77:24:27:8e:c6', '34:79:16:d3:4d:71', 'a2:51:44:ab:a5:e1', '3c:6a:48:18:59:a1', '3a:ca:84:31:a2:73', '18:f2:2c:e3:5a:99', '3c:06:a7:40:c1:4c', '12:5b:ad:5f:1c:52', 'c2:bf:a3:c6:58:a0', '0e:f9:f8:02:90:38', 'aa:93:4a:7f:25:ce', '6a:77:24:27:8e:c4', '16:cb:19:13:97:0a', 'b2:22:7a:80:87:e9', '34:79:16:d3:4d:74', '04:f9:f8:02:90:38', '6a:77:24:27:8e:c6', '12:82:3d:1e:cc:c6', 'f6:84:8d:72:79:f8', '12:82:3d:1e:cc:da', 'c2:69:59:1d:cb:53', '12:82:3d:de:cc:da', '34:79:16:d3:4d:75', '12:82:3d:de:cd:0e', '68:77:24:06:e4:26', '3c:cd:57:f9:3a:41', 'f6:6d:2f:ae:45:28', '90:98:38:f5:d5:ee', '82:6e:f5:3c:f6:c2', 'f6:84:8d:f2:79:f8']
    # 生成对应的特征向量
    ss = []
    for m in mac_list:
        if m in data_dict.keys():
            ss.append(data_dict[m])
        else:
            ss.append(0)
    ss_np = np.array(ss)
    ss_last = torch.tensor(ss_np).to(device='cuda:0')
    ss_last = torch.reshape(ss_last,(1, 95))
    return ss_last

def deal(pred):
    """
    x = x_pre * 0.8 - 23
    y = y_pre * 0.8 - 46
    """
    pred_x,pred_y = pred
    x_pre = torch.max(pred_x, dim=1)[1].item()
    y_pre = torch.max(pred_y, dim=1)[1].item()
    # print(x_pre, y_pre)
    x = x_pre * 0.8 - 23
    y = y_pre * 0.8 - 46
    posi = {"x":x, "y":y}
    return posi

if __name__ == '__main__':
    app.run()

=======
from flask import Flask, request, jsonify
import torch
import numpy as np
from CNN import CNN

app = Flask(__name__)

# Load the trained model
model = torch.load('best_CNN_lalala_1.pth')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from the request
    data = request.json['data']

    # Perform inference using the trained model
    input_data = process(data)
    output = model(input_data)
    pos = deal(output)
     # Return the prediction as a JSON response
    return jsonify("position", pos)

def process(data_dict):
    """
    :param data_dict: 在当前位置收集到的 mac address --> ss 字典
    :return: ss特征向量
    """
    mac_list = ['c2:bf:a3:c6:58:96', 'f6:6d:2f:de:45:28', 'f4:84:8d:c2:79:f8', 'f4:6d:2f:64:94:a1', '34:79:16:f3:4d:74', '3c:cd:57:f9:3a:42', '12:82:3d:8e:cc:ea', '0a:f9:f8:02:90:3a', '12:82:3d:8e:cc:ee', 'f4:6d:2f:64:94:9f', '22:0b:20:bf:b6:b9', '16:cb:19:11:97:c9', 'f8:8c:21:e3:e6:a5', 'f8:8c:21:e3:e6:a7', '2a:3a:4d:04:6c:95', 'f2:a6:54:c0:f8:4c', '54:71:dd:00:5a:99', '90:98:38:45:d5:ec', 'fa:6d:2f:64:94:a1', '12:82:3d:5e:cc:da', '12:82:3d:1e:cc:ee', 'f4:84:8d:c2:79:fa', 'c2:69:59:1d:cb:4f', '18:f2:2c:e3:5a:9e', '58:50:ed:f3:f7:82', 'ca:94:02:fc:66:d5', '12:82:3d:8e:cc:da', '12:82:3d:8e:cd:0e', '64:6e:97:d9:c4:b8', 'c2:bf:a3:c6:58:92', '52:84:92:a1:01:08', 'd4:9e:3b:b2:b2:e5', '0a:f9:f8:02:90:38', '44:df:65:6c:ea:d1', '34:79:16:d3:4d:76', '5c:ea:1d:0f:15:0d', 'f4:2a:7d:f1:13:e7', '12:82:3d:de:cc:ee', '12:82:3d:5e:cd:0e', '90:98:38:f5:d5:ed', 'f4:6d:2f:ee:45:28', '12:82:3d:5e:cc:c6', '42:cd:57:f9:3a:42', '12:82:3d:5e:cc:ee', '04:f9:f8:02:90:3a', '12:82:3d:5e:cc:ea', 'f4:6d:2f:ee:45:2a', '54:71:dd:05:8c:35', '22:0b:20:bf:b6:b5', '46:df:65:4c:ea:d0', 'f8:8c:21:15:3e:e3', '12:82:3d:de:cc:ea', 'fe:6d:2f:64:94:9f', '12:82:3d:1e:cc:ea', '68:77:24:27:8e:c4', '12:82:3d:1e:cd:0e', '02:e8:53:d6:0c:c2', '68:77:24:06:e4:4c', '82:6e:f5:3c:f6:be', '44:df:65:6c:ea:d0', '90:98:38:45:d5:fc', '3e:6a:48:28:59:a1', '90:98:38:45:d5:e8', 'c2:bf:a3:c6:58:a1', '90:98:38:45:d5:f8', '68:77:24:27:8e:c6', '34:79:16:d3:4d:71', 'a2:51:44:ab:a5:e1', '3c:6a:48:18:59:a1', '3a:ca:84:31:a2:73', '18:f2:2c:e3:5a:99', '3c:06:a7:40:c1:4c', '12:5b:ad:5f:1c:52', 'c2:bf:a3:c6:58:a0', '0e:f9:f8:02:90:38', 'aa:93:4a:7f:25:ce', '6a:77:24:27:8e:c4', '16:cb:19:13:97:0a', 'b2:22:7a:80:87:e9', '34:79:16:d3:4d:74', '04:f9:f8:02:90:38', '6a:77:24:27:8e:c6', '12:82:3d:1e:cc:c6', 'f6:84:8d:72:79:f8', '12:82:3d:1e:cc:da', 'c2:69:59:1d:cb:53', '12:82:3d:de:cc:da', '34:79:16:d3:4d:75', '12:82:3d:de:cd:0e', '68:77:24:06:e4:26', '3c:cd:57:f9:3a:41', 'f6:6d:2f:ae:45:28', '90:98:38:f5:d5:ee', '82:6e:f5:3c:f6:c2', 'f6:84:8d:f2:79:f8']
    # 生成对应的特征向量
    ss = []
    for m in mac_list:
        if m in data_dict.keys():
            ss.append(data_dict[m])
        else:
            ss.append(0)
    ss_np = np.array(ss)
    ss_last = torch.tensor(ss_np).to(device='cuda:0')
    ss_last = torch.reshape(ss_last,(1, 95))
    return ss_last

def deal(pred):
    """
    x = x_pre * 0.8 - 23
    y = y_pre * 0.8 - 46
    """
    pred_x,pred_y = pred
    x_pre = torch.max(pred_x, dim=1)[1].item()
    y_pre = torch.max(pred_y, dim=1)[1].item()
    # print(x_pre, y_pre)
    x = x_pre * 0.8 - 23
    y = y_pre * 0.8 - 46
    posi = {"x":x, "y":y}
    return posi

if __name__ == '__main__':
    app.run()

>>>>>>> e9b23a010084673d79b90c3db24313e1ca4a5163
