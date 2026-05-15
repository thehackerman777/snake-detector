# Snake Detector 🐍🐉
Sistema de visión computacional para detección y clasificación de serpientes del maíz (*Pantherophis guttatus*).

## Estado Actual
- ✅ Infraestructura OCI (VM.Standard3.Flex, Object Storage)
- ✅ Servidor FastAPI funcionando
- ✅ 230 imágenes reales de corn snakes (iNaturalist CC-BY)
- ✅ Interfaz de etiquetado web para clasificar morphs
- ⏳ Etiquetado de dataset (proceso manual)
- ⏳ Entrenamiento de modelo YOLOv8n
- ⏳ App móvil KMP

## Acceso
- **Servidor:** http://157.137.233.78
- **Etiquetado:** http://157.137.233.78/label
- **API Health:** http://157.137.233.78/health
- **GitHub:** https://github.com/thehackerman777/snake-detector

## Stack
- **Backend:** FastAPI + Python
- **Infra:** Oracle Cloud (sa-bogota-1)
- **Modelo:** YOLOv8n (pendiente de entrenar)
- **Dataset Fuente:** iNaturalist (CC-BY, research-grade)
