import torch 
from torchvision.transforms import transforms
from torchvision.datasets import MNIST

#MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]) 
trainset = MNIST(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32 , shuffle=True, num_workers=0)
testset = MNIST(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32 , shuffle=False, num_workers=0)
# the classes 
classe = (0,1,2,3,4,5,6,7,8,9)
#create a neural network model 

class DigitClassifier(torch.nn.Module):
    def __init__(self) :
        super( DigitClassifier,self).__init__()
        self.inputlayer = torch.nn.Linear(in_features= 784 ,out_features= 180  )
        self.hiddenlayer1 = torch.nn.Linear (in_features= 180 , out_features= 50)
        self.outputlayer = torch.nn.Linear ( in_features= 50 , out_features= 10)
    def forward(self,X):
        X = torch.nn.functional.relu(self.inputlayer(X))
        X = torch.nn.functional.relu(self.hiddenlayer1(X))
        return torch.nn.functional.softmax(self.outputlayer(X), dim = -1)

#model instance 
model = DigitClassifier()
#define loss function and optimizer
loss_funtion = torch.nn.CrossEntropyLoss()
optimizer_function = torch.optim.Adam(model.parameters(),lr= 0.001 )
#epoch
epoch = 10 
int_running_acc = 0
int_running_loss = 0
# trainning phase
for epoch in range(epoch):

    # initialize loss and precision
    running_loss = 0.0
    running_acc = 0.0
    
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        # initialize gradient
        optimizer_function.zero_grad()
        
        # prediction
        outputs = model(inputs)
        # loss calculation
        loss = loss_funtion(outputs, labels)

        # backward propagation
        loss.backward()

        # update weigths and bias 
        optimizer_function.step()

        # these variables will take loss and accuracy for each 200 lot for intermediate stats
        int_running_loss += loss.item()
        int_running_acc += (outputs.argmax(1) == labels).sum().item()
        #intermediate statistics
        if i % 200 == 199 :
            print('[epoch %d ; lot %d] loss: %.3f, acc: %.3f' %(epoch + 1, i+1, int_running_loss / 200, int_running_acc / (200 * 32)))
            
            int_running_acc = 0
            int_running_loss = 0 
       # these variables will take loss and accuracy for 1 epoch
        running_loss +=  loss.item()
        running_acc +=  (outputs.argmax(1) == labels).sum().item()   

    # statistics for each epoch
    print(" statistic for each epoch ")
    print('[epoch %d] loss: %.3f, acc: %.3f' %(epoch + 1, running_loss / i, running_acc / (i * 32)))
            
print('Finished Training')

# testing precision
test_acc = 0.0

# evaluation mode
model.eval()

# testing phase
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        test_acc += (outputs.argmax(1) == labels).sum().item()

# mean accuracy 
test_acc = test_acc / len(testset)

print('Accuracy of the network on the 10000 test images: %.3f' % test_acc)
print(" a prediction ")
img,label =testset[5]
out = model(img)
print(out.argmax() , label)


