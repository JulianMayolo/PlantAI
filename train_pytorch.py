'''
Projeto de IA: Classificação de Folhas (Saudáveis vs. Doentes)

Objetivo: Desenvolver um modelo de Inteligência Artificial para classificar
imagens de folhas de plantas em duas categorias: 'healthy' (saudável) ou
'diseased' (doente).

Dataset: "Leaves: Healthy or Diseased"
Fonte: https://www.kaggle.com/datasets/prasanshasatpathy/leaves-healthy-or-diseased
O dataset já vem pré-dividido em pastas de treino (train) e validação (val).
Cada uma dessas pastas contém subpastas com os nomes das classes:
- diseased
- healthy

Abordagem: Utilizaremos uma Rede Neural Convolucional (CNN), uma arquitetura
especializada e altamente eficaz para tarefas de classificação de imagens.
'''
import os  # os, time: para interagir com o sistema operacional e medir o tempo.
import time
import torch # torch: framework principal de deep learning.
import torch.nn as nn # nn, optim: para construir a rede e otimizar os pesos.
import torch.optim as optim
from torchvision import datasets, transforms #para manipulação de datasets de imagem e transformações.
from torch.utils.data import DataLoader # DataLoader: para carregar os dados em lotes.
from timm import create_model
from sklearn.metrics import f1_score # sklearn.metrics: para calcular métricas de avaliação como o F1-Score.


def main():
    """Função principal para encapsular todo o processo."""
    
    # Exibe a versão do PyTorch e verifica se uma GPU (CUDA) está disponível.
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # --- 1. PREPARAÇÃO DOS DADOS ---
    # Define os caminhos para as pastas do dataset.
    dataset_path = r'C:\Users\Pichau\Leaves_Heatlhy_IA\leaves\leaves_healthy_or_diseased'
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')

    # Verifica se os caminhos do dataset existem antes de prosseguir.
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("ERRO: Pastas 'train' ou 'val' não encontradas no caminho do dataset.")
        exit()

    # Define uma sequência de transformações a serem aplicadas em cada imagem.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),       # Redimensiona a imagem para 224x224 pixels.
        transforms.ToTensor(),               # Converte a imagem em um tensor PyTorch.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normaliza os pixels.
    ])

    # Carrega os datasets de treino e validação a partir das pastas.
    # ImageFolder assume que as subpastas são as classes.
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    # Cria os DataLoaders para carregar os dados em lotes (batches).
    # O shuffle=True embaralha os dados de treino a cada época.
    # num_workers=0 é uma configuração para evitar erros de multiprocessing no Windows.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Exibe informações sobre os dados carregados.
    class_names = train_dataset.classes
    print(f"Classes encontradas: {class_names}")
    print(f"Total de imagens de treino: {len(train_dataset)}")
    print(f"Total de imagens de validação: {len(val_dataset)}")


    # --- 2. CONFIGURAÇÃO DO MODELO (ViT) ---
    # Define o dispositivo de hardware (GPU se disponível, senão CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cria o modelo Vision Transformer (ViT) usando a biblioteca 'timm'.
    # pretrained=True: carrega os pesos de um modelo já treinado no dataset ImageNet.
    # num_classes: ajusta a camada final do modelo para o nosso número de classes (2).
    model = create_model('vit_base_patch16_224', pretrained=True, num_classes=len(class_names))
    
    # Envia o modelo para o dispositivo selecionado (GPU ou CPU).
    model.to(device)
    print("Modelo Vision Transformer (ViT) carregado.")

    # --- 3. CONFIGURAÇÃO DO TREINAMENTO ---
    # Define a função de perda (loss). CrossEntropyLoss é padrão para classificação.
    criterion = nn.CrossEntropyLoss()
    # Define o otimizador. Adam é um algoritmo eficiente para ajustar os pesos da rede.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- 4. LOOP DE TREINAMENTO ---
    epochs = 10         # Número de vezes que o modelo verá todo o dataset de treino.
    best_val_f1 = 0.0   # Variável para rastrear o melhor F1-Score na validação.

    # Inicia o loop que percorre todas as épocas.
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- FASE DE TREINO ---
        # Ativa o modo de treino no modelo (habilita camadas como dropout).
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        
        # Itera sobre os lotes de dados de treino.
        for images, labels in train_loader:
            # Move as imagens e rótulos para o dispositivo (GPU/CPU).
            images, labels = images.to(device), labels.to(device)
            
            # 1. Zera os gradientes acumulados de iterações anteriores.
            optimizer.zero_grad()
            # 2. Forward pass: faz a previsão do modelo para as imagens de entrada.
            outputs = model(images)
            # 3. Calcula a perda (erro) entre a previsão e os rótulos reais.
            loss = criterion(outputs, labels)
            # 4. Backward pass: calcula os gradientes da perda em relação aos pesos do modelo.
            loss.backward()
            # 5. Atualiza os pesos do modelo usando os gradientes calculados.
            optimizer.step()
            
            # Acumula a perda e as previsões para calcular as métricas da época.
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
        # Calcula a perda e o F1-Score médios para a época de treino.
        train_loss = running_loss / len(train_loader.dataset)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        # --- FASE DE VALIDAÇÃO ---
        # Ativa o modo de avaliação (desabilita camadas como dropout).
        model.eval()
        running_loss = 0.0
        val_preds, val_labels = [], []
        
        # Desativa o cálculo de gradientes para acelerar a validação.
        with torch.no_grad():
            # Itera sobre os lotes de dados de validação.
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        # Calcula a perda e o F1-Score médios para a época de validação.
        val_loss = running_loss / len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        epoch_time = time.time() - start_time
        
        # Exibe os resultados da época.
        print(f"Epoch {epoch+1}/{epochs} | Tempo: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train F1-Score: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f},   Val F1-Score: {val_f1:.4f}")

        # Verifica se o modelo atual é o melhor até agora com base no F1-Score de validação.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Salva os pesos (o "estado") do melhor modelo em um arquivo.
            torch.save(model.state_dict(), 'best_vit_model.pth')
            print(f"  --> Novo melhor modelo salvo com F1-Score: {best_val_f1:.4f}")

    print("\nTreinamento concluído! O melhor modelo foi salvo como 'best_vit_model.pth'.")

# --- PONTO DE ENTRADA DO SCRIPT ---
# Este bloco `if __name__ == '__main__':` garante que a função `main()`
# só será executada quando o arquivo for rodado diretamente (ex: "python train_pytorch.py").
if __name__ == '__main__':
    main()