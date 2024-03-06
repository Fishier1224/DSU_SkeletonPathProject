import os
from src.model import Net,MyDataset,DataLoader
import torch
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
import tensorboard

data_root = 'dataset/taichi24/marked_pic'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train_net_cnn(out_model_name,epoch):
    train_data = MyDataset(txt=data_root + 'train.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=20, shuffle=True)

    model = Net()
    # print(model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()

    plt_loss = []
    plt_acc = []

    writer = SummaryWriter()
    tensorboardMark = True

    for epoch in range(epoch):
        print('epoch {}'.format(epoch + 1))
        train_loss = 0.
        train_acc = 0.
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()

            if tensorboardMark:
                tensorboardMark = False
                with SummaryWriter(comment="Net") as w:
                    w.add_graph(model, (batch_x,))

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        plt_acc.append(train_acc / (len(train_data)))
        plt_loss.append(train_loss / (len(train_data)))

        writer.add_scalar('loss', train_loss / (len(train_data)), epoch)
        writer.add_scalar('acc', train_acc / (len(train_data)), epoch)

        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / len(train_data), train_acc / len(train_data)))

        model.eval()

    # ===================================== 绘图 ==================================
    plt.figure(12, figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.title("network loss")
    plt.plot(plt_loss, 'r')
    plt.subplot(1, 2, 2)
    plt.title("network acc")
    plt.plot(plt_acc, 'g')
    plt.savefig('history/'+out_model_name+'_loss.png')
    plt.close(12)
    # ===================================== 绘图 ==================================

    torch.save(model.state_dict(), "model/"+out_model_name+'.pth')


if __name__ == '__main__':
    # convert_to_img() # 处理图片 写入TXT
    train_net_cnn('24classification_dropout0.8_pic',10)
