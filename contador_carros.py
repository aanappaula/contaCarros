# -*- coding: utf-8 -*-
"""
Contador de Carros com Visão Computacional
Universidade Univille - Unidade 04 - Aprendizado de Máquinas
Aluna: Ana Paula de Souza
TURMA 2026/2

Técnica: Subtração de Fundo (MOG2) + Detecção de Contornos + Linha Virtual
         com rastreamento de centroide para evitar contagem duplicada.
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────────
# PARÂMETROS — ajuste conforme o seu vídeo
# ─────────────────────────────────────────────────
ARQUIVO_VIDEO  = 'video.mp4'
ARQUIVO_SAIDA  = 'resultado_contagem.mp4'
AREA_MINIMA    = 1500  # área mínima (px²) para considerar um carro (1080p é maior)
OFFSET         = 12    # tolerância em pixels ao cruzar a linha
COOLDOWN_FRAMES = 20   # após contar um carro, ignora a mesma região por N frames
DIST_MINIMA    = 80    # distância mínima (px) entre dois carros diferentes
MAX_FRAMES     = 0     # 0 = processa o vídeo inteiro

# ─────────────────────────────────────────────────
# Inicialização
# ─────────────────────────────────────────────────
cap = cv2.VideoCapture(ARQUIVO_VIDEO)

if not cap.isOpened():
    print(f'ERRO: Não foi possível abrir "{ARQUIVO_VIDEO}".')
    exit(1)

largura      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print('=== Informações do Vídeo ===')
print(f'Resolução   : {largura} x {altura} pixels')
print(f'FPS         : {fps:.1f}')
print(f'Total frames: {total_frames}')
print(f'Duração     : {total_frames / fps:.1f} segundos')
print()
print('Pressione  Q  para parar antes do fim.')
print()

LINHA_Y = altura // 2

subtrator = cv2.createBackgroundSubtractorMOG2(
    history=300, varThreshold=60, detectShadows=True
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
saida  = cv2.VideoWriter(ARQUIVO_SAIDA, fourcc, fps, (largura, altura))

contador_carros = 0
frame_idx       = 0

# Dicionário: cx -> frame em que foi contado (para o cooldown)
# Formato: { cx_aproximado: frame_contado }
carros_contados = {}

# ─────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────
def ja_foi_contado(cx, frame_atual):
    """Retorna True se já existe um carro contado próximo a cx recentemente."""
    for cx_anterior, frame_contado in list(carros_contados.items()):
        # Remove entradas antigas (fora do cooldown)
        if frame_atual - frame_contado > COOLDOWN_FRAMES:
            del carros_contados[cx_anterior]
            continue
        if abs(cx - cx_anterior) < DIST_MINIMA:
            return True
    return False

# ─────────────────────────────────────────────────
# Loop principal
# ─────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if MAX_FRAMES > 0 and frame_idx >= MAX_FRAMES:
        break

    # 1. Subtração de fundo
    mascara = subtrator.apply(frame)
    _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)

    # 2. Limpeza morfológica
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.dilate(mascara, kernel, iterations=2)

    # 3. Detecção de contornos
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 4. Linha de contagem (amarela)
    cv2.line(frame, (0, LINHA_Y), (largura, LINHA_Y), (0, 255, 255), 2)

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < AREA_MINIMA:
            continue

        x, y, w, h = cv2.boundingRect(contorno)
        cx = x + w // 2
        cy = y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

        # 5. Verifica cruzamento da linha COM proteção de duplicata
        if (LINHA_Y - OFFSET) < cy < (LINHA_Y + OFFSET):
            if not ja_foi_contado(cx, frame_idx):
                contador_carros += 1
                carros_contados[cx] = frame_idx
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(
                    frame, 'CONTADO', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

    # 6. Placar
    cv2.rectangle(frame, (0, 0), (240, 55), (0, 0, 0), -1)
    cv2.putText(
        frame, f'Carros: {contador_carros}',
        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3
    )

    cv2.imshow('Contador de Carros', frame)
    saida.write(frame)
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Interrompido pelo usuário.')
        break

# ─────────────────────────────────────────────────
# Finalização
# ─────────────────────────────────────────────────
cap.release()
saida.release()
cv2.destroyAllWindows()

print('=' * 45)
print('        RESULTADO FINAL DO CONTADOR')
print('=' * 45)
print(f'  Frames processados : {frame_idx}')
print(f'  Carros contados    : {contador_carros}')
print(f'  Vídeo salvo em     : {ARQUIVO_SAIDA}')
print('=' * 45)
