from model import *
from model import TimeSeriesDataset


config = {
    'n_classes': 3,
}

def get_acc(model_name, dataset_name, mode, seq_len, ratio, num_classes):
    
    config['n_classes'] = num_classes
    
    TSD = TimeSeriesDataset
    CAC = classification_accuracy_sp

    
    # 使用pickle加载模型文件
    loaded_model = pickle.load(open(f"/home/dcz/wp/AAE/AAE_me/autoencoder_lstms/model/{model_name}", "rb"))

    with open(f'{dataset_name}', 'rb') as file:
        dataset = pickle.load(file)

    lenth = len(dataset['data'])
    lenth = lenth - lenth % 32
    N = lenth
    print(f'N: {N}')
    data = dataset['data'][: N]
    labels = dataset['label'][: N]


    BATCH_SIZE = 32
    SEQ_LEN = seq_len
    INPUT_FEATURES_NUM = int(dataset['data'].shape[1] / SEQ_LEN)

    test_data = data
    test_labels = labels
    test_data = np.array(test_data); test_labels = np.array(test_labels)
    test_dataset = TSD(test_data, test_labels, SEQ_LEN, INPUT_FEATURES_NUM, ratio, num_classes=config['n_classes'])

    # 创建 DataLoader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    V_Acc = []
    V_F1 = []
    V_Prec = []
    V_Recall = []
    V_relate_Acc = []
    V_b_Acc = []
    
    for test_data, test_labels in test_loader:
        test_data = test_data.cuda()
        test_labels = test_labels.cuda()
        _, pred_res = loaded_model(test_data)
        acc, f1, recall, prec, racc, b_acc = CAC(pred_res, test_labels, mode='val', num_classes=config['n_classes'], path_len=SEQ_LEN)
        V_Acc.append(acc.item())
        V_F1.append(f1)
        V_Prec.append(prec)
        V_Recall.append(recall)
        V_relate_Acc.append(racc.item())
        V_b_Acc.append(b_acc)
        
    acc = np.mean(V_Acc)

    # f1 = np.mean(V_F1, axis=0)  # 应该不会再抛出错误
    # prec = np.mean(V_Prec, axis=0)
    # recall = np.mean(V_Recall, axis=0)
    racc = np.mean(V_relate_Acc)
    b_acc = np.mean(V_b_Acc)
    # print(f'Accuracy: {acc}')

    return acc, f1, prec, recall, racc, b_acc




if __name__ == '__main__':
    probe_class = 'E'
    topoloy_class = 'T'
    probe_rate = 2
    bins = 9

