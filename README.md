# 🚗 Contador de Carros com Visão Computacional

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-ready-orange?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Univille-2026%2F2-purple?style=for-the-badge"/>
</p>

<p align="center">
  Projeto acadêmico para contagem automática de veículos em vídeo usando visão computacional.
  <br/>
  Universidade Univille · Unidade 04 — Aprendizado de Máquinas · TURMA 2026/2
  <br/>
  <strong>Ana Paula de Souza</strong>
</p>

---

## 📌 Sobre o Projeto

Este projeto detecta e conta automaticamente os carros que passam por uma rua a partir de um vídeo de câmera estática.  
A contagem é feita por uma **linha virtual horizontal** desenhada no centro do frame — sempre que o centroide de um veículo cruza essa linha, o contador incrementa.

O sistema também detecta o **sentido de circulação** (subindo ↑ ou descendo ↓ na tela), o que permite analisar tráfego em ruas de mão dupla.

---

## 🧠 Técnica Utilizada

```
Vídeo → Subtração de Fundo (MOG2) → Limpeza Morfológica → Detecção de Contornos
      → Rastreamento por ID → Cruzamento da Linha → Contagem por Sentido
```

| Etapa | Método |
|---|---|
| Separar fundo do movimento | MOG2 (Mixture of Gaussians) |
| Remover ruído da máscara | Morfologia: Open → Close → Dilate |
| Detectar veículos | `cv2.findContours` com filtro de área mínima |
| Rastrear entre frames | Associação por distância euclidiana do centroide |
| Evitar contagem dupla | Cooldown por ID (bloqueio de N frames após contagem) |
| Detectar sentido | Comparação do `cy` atual vs anterior do mesmo ID |

---

## 🗂️ Estrutura do Projeto

```
contaCarros/
├── contador_carros.py       # Script Python para rodar localmente
├── contador_carros.ipynb    # Notebook para Google Colab
├── video.mp4                # Vídeo de entrada (não versionado)
├── resultado_contagem.mp4   # Vídeo de saída gerado (não versionado)
└── README.md
```

---

## ▶️ Como Usar

### Opção 1 — Localmente (`.py`)

**Pré-requisito:** Python 3.8+ com OpenCV instalado.

```bash
pip install opencv-python numpy
```

Coloque o `video.mp4` na mesma pasta e execute:

```bash
python contador_carros.py
```

Uma janela será aberta mostrando o vídeo processado em tempo real.  
Pressione **Q** para interromper antes do fim.  
O resultado é salvo automaticamente em `resultado_contagem.mp4`.

---

### Opção 2 — Google Colab (`.ipynb`)

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Faça upload do arquivo `contador_carros.ipynb`
3. No painel lateral (ícone de pasta), faça upload do `video.mp4`
4. Execute as células em ordem

---

## ⚙️ Parâmetros Ajustáveis

Localizados no início da seção principal do código:

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `AREA_MINIMA` | `1500` | Área mínima (px²) para considerar um objeto como carro |
| `OFFSET` | `12` | Espessura da zona de detecção ao redor da linha (px) |
| `COOLDOWN_FRAMES` | `20` | Frames de bloqueio após contar um carro (evita duplicatas) |
| `DIST_MINIMA` | `80` | Distância máxima (px) para associar detecção ao mesmo ID |
| `MAX_FRAMES` | `0` | Limite de frames processados (`0` = vídeo inteiro) |

> **Dica:** Se a contagem ficar alta demais, aumente `COOLDOWN_FRAMES`.  
> Se estiver perdendo carros, diminua `AREA_MINIMA` ou aumente `OFFSET`.

---

## 📊 Saída esperada no terminal

```
=== Informações do Vídeo ===
Resolução   : 1920 x 1080 pixels
FPS         : 30.0
Total frames: 360
Duração     : 12.0 segundos

=============================================
        RESULTADO FINAL DO CONTADOR
=============================================
  Frames analisados   : 360
  Total de carros     : 3
  Sentido v (descendo): 2
  Sentido ^ (subindo) : 1
  Vídeo de saída      : resultado_contagem.mp4
=============================================
```

---

## 📚 Referências

- [OpenCV Documentation](https://docs.opencv.org)
- [Background Subtraction — OpenCV](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
- [Cascade Classifier — OpenCV](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
