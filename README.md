# Snake Detector 🐍🔍

Sistema de visión computacional en tiempo real para la detección y clasificación de serpientes del maíz (*Pantherophis guttatus*) y sus variaciones fenotípicas (morphs).

## Arquitectura

```
┌──────────────┐     ┌──────────────────────────┐     ┌──────────────┐
│  Android App │────▶│  FastAPI Backend (OCI)   │────▶│   YOLOv8n   │
│  (KMP)       │◀────│  + WebSocket + REST      │◀────│  (OpenVINO) │
└──────────────┘     └──────────┬───────────────┘     └──────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   PostgreSQL (Docker)  │
                    │   Oracle Object Store  │
                    └───────────────────────┘
```

## Stack

- **Modelo:** YOLOv8n + OpenVINO FP16
- **Backend:** FastAPI + Uvicorn + WebSockets
- **Infra:** Oracle Cloud (VM.Standard3.Flex, 1 OCPU, 4GB RAM)
- **Almacenamiento:** Oracle Object Storage + PostgreSQL
- **Móvil:** Kotlin Multiplatform (Android)
- **CI/CD:** GitHub Actions + Docker
- **Orquestación:** OpenClaw

## Estructura

```
snake-detector/
├── backend/          # API REST + WebSocket server
│   ├── app/          # Código fuente FastAPI
│   ├── docker/       # Dockerfiles
│   └── tests/        # Tests
├── ml-pipeline/      # Entrenamiento y modelos
│   ├── datasets/     # Conjuntos de datos
│   ├── notebooks/    # Jupyter notebooks
│   ├── models/       # Modelos exportados
│   └── scripts/      # Scripts de entrenamiento
└── android-kmp/      # App móvil
```

## Requisitos

- Python 3.11+
- Docker + Docker Compose
- Oracle CLI configurado
- YOLOv8 (Ultralytics)

## Inicio Rápido

```bash
# Backend
cd backend
docker-compose up -d

# ML Pipeline
cd ml-pipeline
pip install -r requirements.txt
python scripts/train.py
```
