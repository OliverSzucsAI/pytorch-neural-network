import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Adat előkészítése
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# Modell
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)  # Első réteg
        self.fc2 = nn.Linear(500, 10)     # Kimeneti réteg

    def forward(self, x):
        x = x.view(-1, 28*28)  # Átalakítjuk a képet vektorrá
        x = torch.relu(self.fc1(x))  # Aktivációs függvény
        x = self.fc2(x)  # Kimeneti réteg
        return x

# Modell inicializálása
net = Net()

# Loss és optimalizáló beállítása
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Modell betanítása
for epoch in range(2):  # Iteráljunk 2 epoch-ra
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # Előre lépés
        outputs = net(inputs)

        # Loss kiszámítása
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:  # Minden 1000 lépés után kiírunk egy statisztikát
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}")
            running_loss = 0.0

print("Befejeződött a tanítás!")

# Tesztelés
correct = 0
total = 0
with torch.no_grad():  # Nem szükséges gradient számítás a tesztelésnél
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Teszt pontosság: {100 * correct / total}%")
