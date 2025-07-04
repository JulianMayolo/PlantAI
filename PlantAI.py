# app_pytorch.py


import os
import torch
import torch.nn as nn
from torchvision import transforms
import google.generativeai as genai
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from timm import create_model

# --- CONFIGURAÇÃO INICIAL ---

# 1. Configurar Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 2. Configurar e Carregar o Modelo PyTorch
device = torch.device("cpu") # Para inferência, CPU geralmente é suficiente
MODEL_PATH = 'best_vit_model.pth'
CLASS_NAMES = ['diseased', 'healthy'] # IMPORTANTE: Na mesma ordem que o ImageFolder (alfabética)

def load_classification_model(model_path):
    try:
        # Recria a arquitetura do modelo
        model = create_model('vit_base_patch16_224', pretrained=False, num_classes=len(CLASS_NAMES))
        # Carrega os pesos que treinamos
        model.load_state_dict(torch.load(model_path, map_location=device))
        # Coloca o modelo em modo de avaliação (importante!)
        model.eval()
        print(f"Modelo PyTorch '{model_path}' carregado com sucesso.")
        return model
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o modelo PyTorch '{model_path}'.")
        print(f"Certifique-se de que o treinamento foi concluído e o arquivo existe.")
        print(f"Erro original: {e}")
        return None

classification_model = load_classification_model(MODEL_PATH)

# Transformação para as imagens de entrada (deve ser a mesma do treino!)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Configurar a API do Gemini (igual ao anterior)
try:
    GEMINI_API_KEY = "AIzaSyDXY81fB65RD16CtsV4C2nju0i41do7myg"
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Modelo Gemini configurado com sucesso.")
except Exception as e:
    gemini_model = None
    print(f"AVISO: Gemini não configurado. {e}")


# --- FUNÇÕES AUXILIARES ---

def classify_leaf_pytorch(image_path):
    if classification_model is None:
        return "Erro no Modelo"
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = inference_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = classification_model(img_tensor)
            _, pred_index = torch.max(outputs, 1)
        
        return CLASS_NAMES[pred_index.item()]
    except Exception as e:
        print(f"Erro ao classificar a imagem {image_path}: {e}")
        return "Erro de Classificação"

def get_gemini_suggestion():
    # Esta função continua igual
    if gemini_model is None:
        return "Modelo Gemini não disponível."
    
    prompt = ("Uma imagem de uma folha de planta foi classificada como 'doente' por um sistema de IA. "
              "Com base nisso, forneça uma resposta geral e prática para um jardineiro amador. "
              "A resposta deve incluir:\n"
              "1. Possíveis causas comuns para doenças em folhas (fungos, pragas, deficiência de nutrientes).\n"
              "2. Primeiros passos recomendados (isolar a planta, remover folhas afetadas).\n"
              "3. Sugestões de tratamentos orgânicos ou de baixa toxicidade (como óleo de neem ou calda de fumo).\n"
              "Seja direto e use uma linguagem simples.")
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Não foi possível gerar a sugestão: {e}"

# --- ROTAS DA APLICAÇÃO WEB (continuam iguais) ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'photos' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('photos')
    healthy_images, diseased_results = [], []
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            status = classify_leaf_pytorch(filepath)
            
            if status == 'healthy':
                healthy_images.append(filename)
            elif status == 'diseased':
                suggestion = get_gemini_suggestion()
                diseased_results.append({'filename': filename, 'suggestion': suggestion})
    return render_template('results.html', healthy_images=healthy_images, diseased_results=diseased_results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- INICIAR A APLICAÇÃO ---
if __name__ == '__main__':
    app.run(debug=True)