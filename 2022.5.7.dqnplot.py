def plot_learning_curve(train_loss, dev_loss, title=''):
    total_steps = len(train_loss)
    x_1 = range(total_steps)
    x_2 = x_1[::len(train_loss) // len(dev_loss)]
    plt.figure(1, figsize=(6, 4))
    plt.plot(x_1, train_loss, c='tab:red', label='train')
    plt.plot(x_2, dev_loss, c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)       
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()      
    plt.show()
 
 
def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
    plt.figure(2, figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()
 
 
def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):   # enumerate() 函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
            writer.writerow([i, p])
 
# 设置存储模型的目录
os.makedirs('my_models', exist_ok=True)
 
# 加载数据
train_data = prep_dataloader('covid.train.csv', 'train', batch_size=135)
dev_data = prep_dataloader('covid.train.csv', 'dev', batch_size=135)
test_data = prep_dataloader('covid.test.csv', 'test', batch_size=135)
 
# 设置模型与训练
mymodel = Mymodel(train_data.dataset.dim)
train_loss, dev_loss = train(mymodel, train_data, dev_data)
plot_learning_curve(train_loss, dev_loss, title='deep model')
del mymodel
 
# 加载最好的模型进行预测
model = Mymodel(train_data.dataset.dim)
ckpt = torch.load('my_models/mymodel.pth', map_location='cpu')  # 加载最好的模型
model.load_state_dict(ckpt)
plot_pred(dev_data, model, 'cpu')
preds = test(model, test_data)
 
# 储存预测结果
save_pred(preds, 'mypred.csv')
print('All Done!')
