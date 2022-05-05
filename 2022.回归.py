
编程的思路可以分成三部分：

①数据的处理

②模型的搭建

③目标的预测

④结果存储与可视化

数据处理部分：

class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 path,
                 mode='train',
                 target_only=False):
        self.mode = mode
 
        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp)) //读入文件
            data = np.array(data[1:])[:, 1:].astype(float)  //去除第一行第一列
