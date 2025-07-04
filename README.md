
# 🌿 PlantAI – Classificador de Folhas (Saudável vs. Doente)

Este projeto utiliza **Inteligência Artificial com PyTorch** e **Vision Transformer (ViT)** para classificar imagens de folhas de plantas em duas categorias:
- **healthy** (saudável)
- **diseased** (doente)

Caso a folha esteja doente, o sistema também gera uma **sugestão de tratamento** utilizando o modelo Gemini (IA generativa do Google).

---

## 📁 Estrutura do Projeto

- `PlantAI.py` – Backend Flask que faz o upload de imagens, classificação via modelo PyTorch e resposta via Gemini.
- `train_pytorch.py` – Script para treinar o modelo ViT usando o dataset de folhas.
- `requirements.txt` – Bibliotecas necessárias.
- `best_vit_model.pth` – Arquivo com os pesos treinados do modelo (**baixar via link abaixo**).
- `templates/` – Contém os arquivos HTML da interface (não incluído aqui).
- `uploads/` – Pasta usada para salvar temporariamente as imagens enviadas pelo usuário.

---

## 🧠 Modelo Treinado

Você pode baixar o modelo treinado (`best_vit_model.pth`) através deste link:

🔗 **[Google Drive - Modelos Treinados](https://drive.google.com/drive/u/0/folders/1gpyL2HIfyIT6sSvN2xfU47gNkr6nYE5d)**

Após o download, salve o arquivo na mesma pasta onde está o `PlantAI.py`.

---

## 🚀 Como Executar

### 1. Clone o projeto:

```bash
git clone https://github.com/SEU_USUARIO/PlantAI.git
cd PlantAI
```

### 2. Crie e ative um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/macOS
```

### 3. Instale as dependências:

```bash
pip install -r requirements.txt
pip install torch torchvision timm flask google-generativeai pillow
```

> ⚠️ O arquivo `requirements.txt` lista dependências mais genéricas. As bibliotecas específicas para o modelo (como `torch`, `timm`, `flask` e `google-generativeai`) precisam ser instaladas manualmente.

---

### 4. Execute a aplicação Flask:

```bash
python PlantAI.py
```

Acesse via navegador: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🏋️‍♂️ Como Treinar o Modelo do Zero

Caso deseje treinar seu próprio modelo:

1. Baixe o dataset:  
   https://www.kaggle.com/datasets/prasanshasatpathy/leaves-healthy-or-diseased

2. Ajuste o caminho do dataset no `train_pytorch.py`.

3. Execute:

```bash
python train_pytorch.py
```

O modelo com melhor desempenho será salvo como `best_vit_model.pth`.

---

## 🧪 Tecnologias Utilizadas

- Python
- PyTorch
- Vision Transformer (ViT)
- Flask
- Google Gemini API (IA Generativa)
- HTML/CSS (templates)
- Pillow (processamento de imagem)

---

Desenvolvido por **Julian Mayolo**  
📍 Projeto educacional para classificação de folhas com IA
