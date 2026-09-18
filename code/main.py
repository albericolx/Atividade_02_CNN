import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os

# Configuração de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')

# Transformações e carregamento do CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Baixando e carregando o dataset CIFAR-10...")
# Usando um subconjunto para treinar mais rápido no ambiente de teste
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainset = torch.utils.data.Subset(full_trainset, range(10000)) # Subconjunto de 10k
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

full_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testset = torch.utils.data.Subset(full_testset, range(2000)) # Subconjunto de 2k
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --- DEFINIÇÃO DOS MODELOS ---

# Modelo Original Base
class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modificação 1: Adição de Dropout
class Mod1CNN(nn.Module):
    def __init__(self):
        super(Mod1CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5) # Dropout adicionado
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Modificação 2: Dropout + LeakyReLU
class Mod2CNN(nn.Module):
    def __init__(self):
        super(Mod2CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
        self.leaky_relu = nn.LeakyReLU(0.1) # Trocado para LeakyReLU

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Modificação 3: Dropout + LeakyReLU + Mais 1 Camada Convolucional
class Mod3CNN(nn.Module):
    def __init__(self):
        super(Mod3CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) # Nova camada
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = self.pool(self.leaky_relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- FUNÇÃO DE TREINAMENTO ---
def train_model(model, name, epochs=3):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n--- Iniciando Treinamento: {name} ---")
    start_time = time.time()
    
    train_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    end_time = time.time()
    train_time = end_time - start_time
    print(f"Treinamento concluído em {train_time:.2f} segundos.")
    
    # Avaliação (Inferência)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Acurácia de Validação: {accuracy:.2f}%")
    
    # Salvar resultados
    model_size = sum(p.numel() for p in model.parameters())
    
    return train_losses, accuracy, train_time, model_size

# --- EXECUÇÃO E COMPARAÇÃO ---
if __name__ == '__main__':
    models = {
        'Original': BaseCNN(),
        'Mod1 (Dropout)': Mod1CNN(),
        'Mod2 (LeakyReLU)': Mod2CNN(),
        'Mod3 (All)': Mod3CNN()
    }
    
    results = {}
    
    for name, model in models.items():
        losses, acc, t_time, size = train_model(model, name, epochs=3)
        results[name] = {'Losses': losses, 'Accuracy': acc, 'Time': t_time, 'Size': size}
        
    # Salvar Gráficos
    plt.figure(figsize=(10,5))
    for name, res in results.items():
        plt.plot(res['Losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Treino)')
    plt.title('Comparativo de Loss por Arquitetura')
    plt.legend()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/loss_comparativo.png')
    print("\nGráfico salvo em results/loss_comparativo.png")
    
    print("\n--- RESUMO COMPARATIVO ---")
    print(f"{'Modelo':<20} | {'Tempo(s)':<10} | {'Acurácia(%)':<12} | {'Parâmetros':<10} | {'Última Loss':<10}")
    print("-" * 75)
    for name, res in results.items():
        print(f"{name:<20} | {res['Time']:<10.2f} | {res['Accuracy']:<12.2f} | {res['Size']:<10} | {res['Losses'][-1]:<10.4f}")
