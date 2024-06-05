# -*- coding:UTF-8 -*-
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Variable
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score

import pickle
import itertools
import warnings
import matplotlib.pyplot as plt

config = {
    "n_classes": 3,
}


# Define CNN
class CnnEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.5):
        super().__init__()
        
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(input_size, num_layers, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_layers, hidden_size, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.cat = nn.Sequential(
            nn.Linear(hidden_size, config['n_classes']),
            nn.Softmax(dim=1)
        )

        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)  # Rearrange dimensions to (batch, channels, seq_len)
        conv_out = self.conv1d_layers(x)
        conv_out = conv_out.permute(0, 2, 1)  # Rearrange back to (batch, seq_len, channels)
        last_hidden = conv_out[:, -1, :]  # Select the output of the last sequence step
        out = self.dropout(last_hidden)
        out = self.cat(out)

        x_ = self.forwardCalculation(conv_out)
        return x_, out

# Define MLP
class MlpEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super().__init__()
        
        # 只使用一个线性变换层
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, config['n_classes']),
            nn.Softmax(dim=1)  # 使用 dim=1 适用于 nn.Sequential 的 Softmax
        )
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):

        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)
        
        x_out = self.mlp(x)
        # print('x', x.shape, batch_size, seq_len)
        # print('x_out', x_out.shape, batch_size, seq_len)
        
        out = self.classifier(x_out)  # Apply classifier to the last time step's output
        x_ = self.forwardCalculation(x_out)  # Apply forward calculation across all outputs
        
        # x_ 降维
        
        return x_, out

class MlpDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super().__init__()
        
        # 定义解码器的MLP结构
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # 最终的输出层将hidden_size变换回原始的output_size
        self.reconstructor = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.mlp(x)
        reconstructed_output = self.reconstructor(x)
        # 使用view来重新形状化输出
        reconstructed_output = reconstructed_output.view(x.size(0), 1, -1)
        return reconstructed_output
        



# Define LSTM Neural Networks

class LstmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.5):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.cat = nn.Sequential(
            nn.Linear(hidden_size, config['n_classes']),
            nn.Softmax(dim=1)  # Ensure you use dim=1 for Softmax with nn.Sequential
        )

        self.forwardCalculation = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        b, s, h = lstm_out.shape  # x is output, size (batch, seq_len, hidden_size)
        # print('b, s, h:', b, s, h)
        out = self.dropout(hn[-1])
        out = self.cat(out)
        
        x_ = self.forwardCalculation(lstm_out)
        return x_, out

    
class LstmDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_rate=0.5):
        super().__init__()
        
        # Assuming the input_size for the decoder is the output_size of the encoder
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        # Output size matches the original feature size of the encoder input
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size) from the encoder
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Applying a linear layer to each time step
        out = self.fc(lstm_out)
        return out

    
class Discriminator_z(nn.Module):
    def __init__(self, input_size):
        super(Discriminator_z, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    # validity 为概率
    def forward(self, z):
        validity = self.model(z)
        return validity
    
class Discriminator_category(nn.Module):
    def __init__(self, input_size):
        super(Discriminator_category, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity
    
    
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, seq_len, features_num, ratio, num_classes=10):
        """
        data: 形状为 (1500, 600) 的张量，包含所有样本。
        labels: 形状为 (1500,) 的张量，包含每个样本的标签。
        num_classes: 类别总数，用于 one-hot 编码。
        """
        self.data = data
        self.labels = labels
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.features_num = features_num
        self.ratio = ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = np.array(self.data[idx]).reshape(self.seq_len, self.features_num)  # 使用.reshape()重塑数组
        
        reduced_size = int(sample.shape[-1] * self.ratio)
        reduced_array = sample[..., :reduced_size]
        sample = torch.tensor(reduced_array, dtype=torch.float32)
        
        label = self.labels[idx]
        # 转换标签为 one-hot 编码
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[label] = 1
        return sample, one_hot_label
    

def classification_accuracy_sp(pred_res, train_labels, mode, num_classes=config['n_classes'], path_len=None):
    # Convert softmax probabilities to class indices
    # Convert softmax probabilities to class indices
    pred_classes = torch.argmax(pred_res, dim=-1)

    # Convert one-hot encoded true labels to class indices
    true_classes = torch.argmax(train_labels, dim=-1)

    # Convert tensors to numpy arrays for sklearn compatibility
    pred_classes_np = pred_classes.cpu()
    true_classes_np = true_classes.cpu()

    pred_t_np = np.zeros((pred_classes_np.shape[0], 1))
    true_t_np = np.zeros((true_classes_np.shape[0], 1))
    for i in range(pred_classes_np.shape[0]):
        if pred_classes_np[i] > 1:
            pred_t_np[i] = 2
        else:
            pred_t_np[i] = pred_classes_np[i]
        
        if true_classes_np[i] > 1:
            true_t_np[i] = 2
        else:
            true_t_np[i] = true_classes_np[i]


    
    # print('pred_classes_np', pred_classes_np[:10], 'true_classes_np', true_classes_np[:10])
    # 如果pred_classes中缺少某种类别，则添加该类别
    for i in range(num_classes):
        if i not in true_classes_np:
            pred_classes_np[-i] = i
            true_classes_np[-i] = i
            
    # Calculate Accuracy
    correct_predictions = (pred_classes == true_classes).sum()
    total_predictions = pred_classes.nelement()
    accuracy = 100. * correct_predictions.float() / total_predictions
    
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    balanced_accuracy = balanced_accuracy_score(true_classes_np, pred_classes_np, adjusted=True) * 100.

    precision_per_class = precision_score(true_t_np, pred_t_np, average=None, zero_division=1)
    recall_per_class = recall_score(true_t_np, pred_t_np, average=None, zero_division=1)
    f1_per_class = f1_score(true_t_np, pred_t_np, average=None, zero_division=1)

    # print('precision_per_class', precision_per_class, 'recall_per_class', recall_per_class, 'f1_per_class', f1_per_class)
    # 计算混淆矩阵
    if mode == 'train':
        
        return accuracy, balanced_accuracy
    
    elif mode == 'val' and num_classes >= 3:
        not_eq_index = np.where((pred_classes_np != true_classes_np) & (true_classes_np >= 2))
        not_eq_index = not_eq_index[0]
        r_acc_sum = 0
        for i in not_eq_index:
            r_err = (abs(pred_classes_np[i] - true_classes_np[i]) / (num_classes)) ** 0.5
            relative_accuracy = 1 - r_err
            r_acc_sum += relative_accuracy
            
        relative_accuracy = (correct_predictions + r_acc_sum) / total_predictions
        
        return accuracy, f1_per_class, recall_per_class, precision_per_class, relative_accuracy, balanced_accuracy
    
    else:
        relative_accuracy = 0
        
        return accuracy, f1_per_class, recall_per_class, precision_per_class, relative_accuracy, balanced_accuracy

def sample_categorical(batch_size, state, probabilities, n_classes=10):
    '''
     Sample from a categorical distribution
     of size batch_size and # of classes n_classes.
     If n_classes is 3 and sampled cat > 2, set cat to 2.
     return: torch.Tensor with the sample
    '''
    # Sample from the categorical distribution
    cat_ = np.random.choice(state.shape[0], size=batch_size, p=probabilities)
    cat = state[cat_]
    
    # If n_classes is 3, modify cat values greater than 2 to be 2
    if n_classes == 3:
        cat = np.where(cat > 2, 2, cat)

    # One-hot encode the cat variable
    cat = np.eye(n_classes)[cat].astype('float32')

    cat = torch.from_numpy(cat) 
    cat = cat.squeeze(1)
    return cat

def return_state_probability(arr):
    # 将每个子数组转换为元组，因为numpy数组是不可哈希的，而元组是可哈希的
    tuples = [tuple(row) for row in arr]
    # 计算每种元组出现的次数
    counts = Counter(tuples)
    # 计算概率：每种元组出现的次数除以总数
    total_count = len(arr)
    probabilities_ = {k: v / total_count for k, v in counts.items()}
    state = np.array(list(probabilities_.keys()))
    probabilities = np.array(list(probabilities_.values()))
    return state, probabilities

def make_singal_Y(N_classes):
    # 路由矩阵
    A_rm = [
        [1 for _ in range(N_classes-1)],
    ]
    # 灯泡总数
    n_bulbs = N_classes - 1
    # 每个灯泡亮起的概率
    P_lights = np.array([0.5 for i in range(n_bulbs)])
    # 抽样数量
    batch_size = 6666

    # 初始化一个数组来存储每个批次的结果
    samples = np.zeros((batch_size, n_bulbs), dtype=int)

    # 对每个样本，独立地决定每个灯泡是否亮起
    for i in range(batch_size):
        # 生成随机数，与灯泡亮起的概率进行比较
        random_values = np.random.rand(n_bulbs)
        # 灯泡亮起条件，比较随机数与概率
        samples[i] = (random_values < P_lights).astype(int)

    # A_rm 矩阵乘 samples矩阵
    A_rm = np.array(A_rm)

    Y = np.dot(A_rm, samples.T) 
    Y = Y.T

    SINGLE_Y = []
    SINGLE_Y = np.array([[Y[i][j]] for i in range(1, Y.shape[0]) for j in range(Y.shape[1])])
    MULTI_Y = Y.copy()

    state_s, prob_s = return_state_probability(SINGLE_Y)
    return state_s, prob_s
            
    
def get_model_single(model_name, dataset_name, seq_len, ratio, num_classes=10, use_prior=True):
    
    config['n_classes'] = num_classes
    
    with open(f'{dataset_name}', 'rb') as file:
        dataset = pickle.load(file)
        
    lenth = len(dataset['data'])

    # lenth = 900

    lenth = lenth - lenth % 64
    N = lenth
    data = dataset['data'][:N]
    labels = dataset['label'][:N]
    STATE, PROB = make_singal_Y(config['n_classes'])
    
    # 打乱数据
    indices = np.arange(N)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]


    Rectruct_Loss = []
    Pred_loss = []
    T_Acc_classes = []
    V_Acc_classes = []

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    train_data_ratio = 0.8
    train_data_len = int(labels.shape[0] * train_data_ratio)
    train_data_len_ = train_data_len
    if not use_prior:
        train_data_len = int(train_data_len * 0.2)
        
    train_data = data[:train_data_len]
    train_labels = labels[:train_data_len]

    BATCH_SIZE = 64
    SEQ_LEN = seq_len
    INPUT_FEATURES_NUM = int(data.shape[1] / SEQ_LEN)
    
    LATEN_FEATURES_NUM = 20
    OUPUT_FEATURES_NUM = 1
    print('train_d_l_', train_data_len_, 'train_d_l', train_data_len)
    test_data = data[train_data_len_:]
    test_labels = labels[train_data_len_:]

    train_data = np.array(train_data); train_labels = np.array(train_labels)
    test_data = np.array(test_data); test_labels = np.array(test_labels)

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_data, train_labels, SEQ_LEN, INPUT_FEATURES_NUM, ratio, num_classes=config['n_classes'])
    test_dataset = TimeSeriesDataset(test_data, test_labels, SEQ_LEN, INPUT_FEATURES_NUM, ratio, num_classes=config['n_classes'])

    # 更新input_features_num
    INPUT_FEATURES_NUM = int(INPUT_FEATURES_NUM * ratio)
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


    # ------ model Prepare ------
    encoder, decoder = None, None
    if seq_len != 1:
        # encoder = LstmEncoder(input_size=INPUT_FEATURES_NUM, hidden_size=64, output_size=LATEN_FEATURES_NUM, num_layers=3)
        encoder = LstmEncoder(input_size=INPUT_FEATURES_NUM, hidden_size=64, output_size=LATEN_FEATURES_NUM, num_layers=2)
        decoder = LstmDecoder(input_size=LATEN_FEATURES_NUM, hidden_size=64, output_size=INPUT_FEATURES_NUM, num_layers=2)
    else:
        print('seq_len == 1')
        encoder = MlpEncoder(input_size=INPUT_FEATURES_NUM * SEQ_LEN, hidden_size=64, output_size=LATEN_FEATURES_NUM)
        decoder = MlpDecoder(input_size=LATEN_FEATURES_NUM, hidden_size=64, output_size=INPUT_FEATURES_NUM * SEQ_LEN)

    discriminator = Discriminator_z(input_size=LATEN_FEATURES_NUM)
    discriminator_cat = Discriminator_category(input_size=config['n_classes'])

    restruct_loss = nn.MSELoss()
    adversarial_loss = nn.BCELoss()

    optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.001)
    optimizer_E_semi = torch.optim.Adam(encoder.parameters(), lr=0.001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    optimizer_D_cat = torch.optim.Adam(discriminator_cat.parameters(), lr=1e-4)

    cuda = torch.cuda.is_available()
    if cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        discriminator.cuda()
        discriminator_cat.cuda()
        adversarial_loss.cuda()
        restruct_loss.cuda()


    # ------------ training ------------
    epoch_times = 200


    for epoch in range(epoch_times):
        encoder.train()
        decoder.train()
        ACC_epoch = []
        class_loss = None
        
        for train_data, train_labels in train_loader:
            if cuda:
                train_data = train_data.cuda()
                train_labels = train_labels.cuda()

            un_labeled = False if random.uniform(0, 1) > 0.2 else True
            if un_labeled == False:
                
                # 设置 真实样本 和 生成样本的 对比标签
                if seq_len != 1:
                    valid = Variable(Tensor(train_data.shape[0], SEQ_LEN, 1).fill_(1.0), requires_grad=False)
                    fake =  Variable(Tensor(train_data.shape[0], SEQ_LEN, 1).fill_(0.0), requires_grad=False)
                else:
                    valid = Variable(Tensor(train_data.shape[0], 1).fill_(1.0), requires_grad=False)
                    fake =  Variable(Tensor(train_data.shape[0], 1).fill_(0.0), requires_grad=False)
                valid_c = Variable(Tensor(train_data.shape[0], 1).fill_(1.0), requires_grad=False)
                fake_c =  Variable(Tensor(train_data.shape[0], 1).fill_(0.0), requires_grad=False)
                
                # 1) Train the autoencoder
                optimizer_G.zero_grad()
                fake_z, fake_cat = encoder(train_data)
                decoded_data = decoder(fake_z)
                
                loss = restruct_loss(train_data, decoded_data)
                loss.backward()
                optimizer_G.step()
                
                # 2) Train the discriminator_z
                if use_prior and seq_len != 1:
                    optimizer_D.zero_grad()
                    real_z = Variable(Tensor(np.random.normal(0, 1, (train_data.shape[0], SEQ_LEN, LATEN_FEATURES_NUM))), requires_grad = False)
                    # print('real_z', real_z.shape, fake_z.shape, train_data.shape[0])
                    real_loss = adversarial_loss(discriminator(real_z), valid)
                    fake_loss = adversarial_loss(discriminator(fake_z.detach()), fake)
                    
                    D_loss = 0.5 * (real_loss + fake_loss)
                    D_loss.backward()
                    optimizer_D.step()
                    
                    # 3) Train the discriminator_category
                    optimizer_D_cat.zero_grad()
                    real_cat = nn.Parameter(sample_categorical(train_data.shape[0], state=STATE, probabilities=PROB, n_classes=config['n_classes']), requires_grad=False).cuda()
                    # print('real_cat', real_cat.shape, fake_cat.shape, valid_c.shape, fake_cat.detach().shape, discriminator_cat(fake_cat.detach()).shape)
                    real_cat_loss = adversarial_loss(discriminator_cat(real_cat), valid_c)
                    fake_cat_loss = adversarial_loss(discriminator_cat(fake_cat.detach()), fake_c)
                    D_cat_loss = 0.5 * (real_cat_loss + fake_cat_loss)
                    D_cat_loss.backward()
                    optimizer_D_cat.step()
                    
                    Rectruct_Loss.append(loss.item())
                # if epoch % 100 == 0:
                #     print('epoch [{}/{}], R_loss:{:.4f}'.format(epoch + 1, epoch_times, loss.data))
                
            else:
                optimizer_E_semi.zero_grad()
                # print("semi-train_data.shape", train_data.shape)
                _, pred_res = encoder(train_data)
                # print('pred_res', pred_res.shape, train_labels.shape)
                # class_loss = F.nll_loss(pred_res, torch.max(train_labels, 1)[1])
                class_loss = F.cross_entropy(pred_res, train_labels)
                class_loss.backward()
                optimizer_E_semi.step()
            
                Pred_loss.append(class_loss.item())
                acc, *_ = classification_accuracy_sp(pred_res, train_labels, mode='train')
                # print('acc', acc.item())
                ACC_epoch.append(acc.item())
            
        ACC_epoch = np.array(ACC_epoch)  
        T_Acc_classes.append(np.mean(ACC_epoch))
        if epoch % 20 == 0:
            print('epoch [{}/{}], Acc:{:.2f}'.format(epoch + 1, epoch_times, np.mean(ACC_epoch)))  

        # ------------ testing ------------
        encoder.eval()
        V_Acc = []
        for test_data, test_labels in test_loader:
            test_data = test_data.cuda()
            test_labels = test_labels.cuda()
            _, pred_res = encoder(test_data)
            acc, *_ = classification_accuracy_sp(pred_res, test_labels, mode='train')
            V_Acc.append(acc.item())
        acc = np.mean(V_Acc)
        V_Acc_classes.append(acc)
        if epoch > epoch_times * 0.8:
            if acc > max(V_Acc_classes):
                pickle.dump(encoder, open(f"/home/dcz/wp/AAE/AAE_me/autoencoder_lstms/model/{model_name}", "wb")) 
            elif (V_Acc_classes[-2] - V_Acc_classes[-1]) > 0.01:
                pickle.dump(encoder, open(f"/home/dcz/wp/AAE/AAE_me/autoencoder_lstms/model/{model_name}", "wb")) 
            else:
                pickle.dump(encoder, open(f"/home/dcz/wp/AAE/AAE_me/autoencoder_lstms/model/{model_name}", "wb")) 
            
                # 
        if epoch % 20 == 0:
            print('epoch [{}/{}], Val Acc:{:.2f}'.format(epoch + 1, epoch_times, acc))
            
    # plt ACC_epoch and V_Acc
    x = range(epoch_times)
    if hasattr(__builtins__,"__IPYTHON__"):
        plt.figure()
        plt.plot(x, T_Acc_classes, label='Train Acc')
        plt.plot(x, V_Acc_classes, label='Val Acc')
        plt.legend()
        plt.show()

    # plt loss
    
    return True
