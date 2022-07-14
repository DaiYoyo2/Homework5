import torch 
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import jieba

img_path =r"实验五数据\data\1.jpg"
img = plt.imread(img_path)
print(type(img))
# ax = fig.add_subplot(111)
plt.imshow(img)
#plt.show()    



counts=[]

#pics=torch.tensor(pics) #图片大小不一致，不能这么干


words=[]
all_words=[]
for i in range(0,5129):
    
    try:
        text=open(r"实验五数据\data\\" + str(i+1) + ".txt",encoding="utf-8")
        try:
            line=text.read()
            line=jieba.lcut(line)
        except:
        
            line=[]
    
    
        counts.append(i+1)
        words.append(line)
        all_words.extend(line)
    except:
        pass

all_words=list(set(all_words))
words_vec=[]
for i in range(0,len(words)):
    words_vec.append([])
    for j in all_words:
        
        if j in words[i]:
            words_vec[i].append(1)
        else:
            words_vec[i].append(0)
    words_vec[i]=np.array(words_vec[i])
    t2=np.zeros(300000)
    words_vec[i]=np.append(words_vec[i],t2)
    words_vec[i]=words_vec[i][0:216284]
words_vec=np.array(words_vec)
print(words_vec.shape)

'''words_vec=list(words_vec)
print(words_vec)'''
x=[]
#print(words_vec[3].shape,pics[3].shape)
#print(np.append(words_vec[2],pics[2][0]))
for i in range(0,len(words)):
    t=[]
    t.append(words_vec[i])
    t=np.array(t)
    x.append(t)
    
print(x[0].shape,len(x))


x=np.array(x)
x=torch.tensor(x)

print(x.shape)

y=[]
text=open(r"实验五数据\train.txt")
line=text.readlines()
line.pop(0)

lable=[]
for i in range(0,len(line)):
    lable.append([])
    t=line[i].split(",")
    lable[i].append(int(t[0]))
    if t[1][0]=="p":

        lable[i].append(1)#积极
    if t[1][2]=="g":
        lable[i].append(2)#消极
    if t[1][2]=="u":
        lable[i].append(0)#无
lable.sort()
lable=np.array(lable)

text=open(r"实验五数据\test_without_label.txt")
line=text.readlines()
line.pop(0)

lable_test=[]
for i in range(0,len(line)):
    lable_test.append([])
    t=line[i].split(",")
    lable_test[i].append(int(t[0]))
    lable_test[i].append(0)
lable_test.sort()
lable_test=np.array(lable_test)
x2=[]
y2=[]
for i in lable:
        t1=i[0]
    
        index_=counts.index(t1)
        
        x2.append(np.array(x[index_]))
        y2.append(i[1])
x_test=[]
y_test=[]
for i in lable_test:
        t1=i[0]
        
        index_=counts.index(t1)
        
        x_test.append(np.array(x[index_]))
        y_test.append(i[1])

x3=np.array(x2)
x2=torch.tensor(x3)
print(x2.shape)
y2=np.array(y2)
y2=torch.tensor(y2)
print(y2.shape)
x_test=np.array(x_test)
x_test=torch.tensor(x_test)
print(x_test.shape)
y_test=np.array(y_test)
y_test=torch.tensor(y_test)
print(y_test.shape)




print(x[0])
# 先转换成 torch 能识别的 Dataset
train_x=x2[0:3800]
train_y=y2[0:3800]
ver_x=x2[3800:4000]
ver_y=y2[3800:4000]
test_dataset=data.TensorDataset(x_test,y_test)
train_dataset = data.TensorDataset(train_x, train_y)
ver_dataset=data.TensorDataset(ver_x, ver_y)
# 把 dataset 放入 DataLoader
test_loader = data.DataLoader(test_dataset,batch_size=1000,shuffle=False)
train_loader = data.DataLoader(train_dataset,batch_size=32,shuffle=False)
ver_loader=data.DataLoader(ver_dataset,batch_size=100,shuffle=False)
import torch.nn as nn
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 5)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(6, 1, 5)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(54068, 1000)
        #self.fc1 = nn.Linear(2497, 120)
        self.fc2 = nn.Linear(1000, 100)
        
        self.fc3 = nn.Linear(100, 3)

    def forward(self, x):
        
        
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.view(x.shape[0], -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
model = LeNet()
print(model)
loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.0005)
loss_count = []
for epoch in range(2):
    for i,(x,y) in enumerate(train_loader):
        
        batch_x = Variable(x) # torch.Size([128, 1, 28, 28])
        batch_y = Variable(y) # torch.Size([128])
        
        # 获取最后输出
        t=batch_x.float()
        out = model(t) # torch.Size([128,10])
        # 获取损失
        t2=batch_y.long()
        loss = loss_func(out,t2)
        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward() # 误差反向传播，计算参数更新值
        opt.step() # 将参数更新值施加到net的parmeters上
        if i%10 == 0:
            loss_count.append(loss)
            print('{}:\t'.format(i), loss.item())
        
    for a,b in ver_loader:
                
                ver_x = Variable(a)
                ver_y = Variable(b)
                t=ver_x.float()
                out = model(t)
                # print('test_out:\t',torch.max(out,1)[1])
                # print('test_y:\t',test_y)
                accuracy = torch.max(out,1)[1].numpy() == ver_y.numpy()
                print('accuracy:\t',accuracy.mean())
            
            
        
    
        
#torch.save(model,r"model2")

for x,y in test_loader:
    
    batch_x = Variable(x)
    t=batch_x.float()
    
    final=model(t)
final=final.detach().numpy()
l=[]
for i in final:
    l.append(np.argmax(i))
to_write=[]
to_write.append("guid,tag\n")
print(len(lable_test))
print(lable_test)

for i in range(0,len(lable_test)):
        
        to_write.append(str(lable_test[i][0]))
        to_write.append(",")
        if l[i]==0:
            to_write.append("neutral")
        if l[i]==1:
            to_write.append("positive")
        if l[i]==2:
            to_write.append("negative")
        to_write.append("\n")
    
output=open("reslut.txt","w")
output.writelines(to_write)
output.close()

#验证准确率56%
#文本64%
