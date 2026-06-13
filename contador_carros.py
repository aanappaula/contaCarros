# -*- coding: utf-8 -*-
"""
Contador de Carros com Visão Computacional
Universidade Univille - Unidade 04 - Aprendizado de Máquinas
TURMA 2026/2

Tecnica: Subtracao de Fundo (MOG2) + Deteccao de Contornos + Linha Virtual
         Rastreamento por ID para evitar contagem duplicada.
"""

import cv2
import numpy as np

# -------------------------------------------------
# PARAMETROS
# -------------------------------------------------
ARQUIVO_VIDEO   = 'video.mp4'
ARQUIVO_SAIDA   = 'resultado_contagem.mp4'
AREA_MINIMA     = 1500   # px minimos para ser considerado um carro
OFFSET          = 12     # tolerancia em pixels ao cruzar a linha
COOLDOWN_FRAMES = 20     # frames de bloqueio apos contar um carro
DIST_MINIMA     = 80     # distancia maxima para associar ao mesmo carro
MAX_FRAMES      = 0      # 0 = processa tudo

# -------------------------------------------------
# Inicializacao
# -------------------------------------------------
cap = cv2.VideoCapture(ARQUIVO_VIDEO)

if not cap.isOpened():
    print('ERRO: Nao foi possivel abrir "' + ARQUIVO_VIDEO + '".')
    exit(1)

largura      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print('=== Informacoes do Video ===')
print('Resolucao   : ' + str(largura) + ' x ' + str(altura) + ' pixels')
print('FPS         : ' + str(round(fps, 1)))
print('Total frames: ' + str(total_frames))
print('Duracao     : ' + str(round(total_frames / fps, 1)) + ' segundos')
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
proximo_id      = 0
rastreados      = {}  # { id: {'cx': x, 'cy': y} }
contados        = {}  # { id: frame_em_que_foi_contado }

# -------------------------------------------------
# Funcoes auxiliares
# -------------------------------------------------
def encontrar_id_proximo(cx, cy, rastreados, dist_max=80):
    melhor_id, melhor_dist = None, dist_max
    for obj_id, pos in rastreados.items():
        dist = np.hypot(cx - pos['cx'], cy - pos['cy'])
        if dist < melhor_dist:
            melhor_dist, melhor_id = dist, obj_id
    return melhor_id

def cooldown_ativo(obj_id, frame_atual):
    return obj_id in contados and (frame_atual - contados[obj_id]) <= COOLDOWN_FRAMES

# -------------------------------------------------
# Loop principal
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if MAX_FRAMES > 0 and frame_idx >= MAX_FRAMES:
        break

    # 1. Subtracao de fundo
    mascara = subtrator.apply(frame)
    _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)

    # 2. Limpeza morfologica
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.dilate(mascara, kernel, iterations=2)

    # 3. Contornos
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 4. Linha de contagem (amarela)
    cv2.line(frame, (0, LINHA_Y), (largura, LINHA_Y), (0, 255, 255), 2)

    ids_atuais = {}

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < AREA_MINIMA:
            continue

        x, y, w, h = cv2.boundingRect(contorno)
        cx = x + w // 2
        cy = y + h // 2

        obj_id = encontrar_id_proximo(cx, cy, rastreados)
        if obj_id is None:
            obj_id = proximo_id
            proximo_id += 1
            rastreados[obj_id] = {'cx': cx, 'cy': cy}

        ids_atuais[obj_id] = {'cx': cx, 'cy': cy}

        cor = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
        cv2.circle(frame, (cx, cy), 6, cor, -1)
        cv2.putText(frame, 'ID ' + str(obj_id), (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2)

        # 5. Verifica cruzamento
        if (LINHA_Y - OFFSET) < cy < (LINHA_Y + OFFSET):
            if not cooldown_ativo(obj_id, frame_idx):
                contador_carros += 1
                contados[obj_id] = frame_idx
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(frame, 'CONTADO', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    rastreados = ids_atuais

    # 6. Placar
    cv2.rectangle(frame, (0, 0), (240, 55), (0, 0, 0), -1)
    cv2.putText(frame, 'Carros: ' + str(contador_carros),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    cv2.imshow('Contador de Carros', frame)
    saida.write(frame)
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Interrompido pelo usuario.')
        break

# -------------------------------------------------
# Finalizacao
# -------------------------------------------------
cap.release()
saida.release()
cv2.destroyAllWindows()

print('=' * 45)
print('        RESULTADO FINAL DO CONTADOR')
print('=' * 45)
print('  Frames processados : ' + str(frame_idx))
print('  Carros contados    : ' + str(contador_carros))
print('  Video salvo em     : ' + ARQUIVO_SAIDA)
print('=' * 45)
