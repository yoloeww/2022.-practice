# 这一模块需要引入的库：
import csv
import numpy 
import tensor
from tensor.utils.data import Dataset, DataLoader
 
class Covid19dataset(Dataset):
    def __init__(self, path, mode='train'):
        super().__init__()
        self.mode = mode
 
        # 读取数据
        with open(path) as file:
            data_csv = list(csv.reader(file))   # 将csv文件中获取的数据转换为列表类型
            data = np.array(data_csv[1:])[:, 1:].astype(float)  # 获取文件中的【数值数据】
 
        if mode == 'test':  # 由于训练集、验证集与测试集的数据有所不同（最后一列数据），在此分情况运行
            data = data[:, 0:93]    # 测试集中，由于最后一行是得预测的数据，故只有93列
            self.data = torch.FloatTensor(data)     # 将numpy类型数据转换为tensor类型
        else:
            target = data[:, -1]    # 训练集和验证集中用于对比结果的“目标”
            data = data[:, 0:93]
            train_index = []
            dev_index = []
            for i in range(data.shape[0]):     # 这个循环用于将covid.train.csv文件中的数据分为训练集和验证集
                if i % 10 != 0:                # 取序号为整十数的样本作为验证集
                    train_index.append(i)
                else:
                    dev_index.append(i)
            if mode == 'train':     # 训练集的数据
                self.target = torch.FloatTensor(target[train_index])
                self.data = torch.FloatTensor(data[train_index, 0:93])
            else:       # 测试集的数据
                self.target = torch.FloatTensor(target[dev_index])
                self.data = torch.FloatTensor(data[dev_index, 0:93])
 
        # 此处是对数据进行标准化处理，可以将不同量纲的不同特征，变为同一个数量级，使得损失函数更加平滑
        # 标准化的优点：①提升模型的精度   ②提升收敛速度
        # 采用均值标准化： （第i维数据 - 第i维数据的平均值）/（第i维数据的标准差）
        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0)) / self.data[:, 40:].std(dim=0)
 
        self.dim = self.data.shape[1]   # 获取数据的列数
 
    def __getitem__(self, item):        # 【注】这是Dataset必须重写的类函数，意在按索引返回数据
        if self.mode == 'train' or self.mode == 'dev':  # 训练集和验证集包含特征和目标数据
            return self.data[item], self.target[item]
        else:
            return self.data[item]      # 测试集仅含有特征数据
 
    def __len__(self):                  # 【注】返回数据的行数
        return len(self.data)
 
 
def prep_dataloader(path, mode, batch_size, n_jobs=0):
    dataset = Covid19dataset(path, mode)    # 定义一个Dataset类
    dataloader = DataLoader(dataset, batch_size,
                            shuffle=(mode == 'train'),  # 【是否打乱数据后再读取】
                            drop_last=False,            # 【False表示不丢弃不能被batch_size整除的部分】
                            num_workers=n_jobs,         # 【采用多少个进程读取数据】
                            pin_memory=False)           # 【是否将数据载入CUDA的内存当中】
    print(mode, 'data done!')
    return dataloader

