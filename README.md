# Video-Relighting-CG

Proyecto orientado a la **inserción realista de una persona en un escenario diferente**, mediante segmentación por green-screen, análisis de iluminación y *relighting*.

## Pipeline

```text
Video de entrada
(Persona + Green Screen)
        │
        ▼
Segmentación y extracción
del sujeto
        │
        ▼
Máscara / Alpha
        │
        ├───────────────┐
        │               │
        ▼               ▼
Persona aislada    Escenario
                  (imagen/video)
                        │
                        ▼
              Estimación de iluminación
                        │
                        ▼
                  Relighting
                        │
                        ▼
                   Composición
                        │
                        ▼
                   Video final
```

## Componentes principales

* **Computer Vision:** segmentación del green screen y generación/refinamiento de la máscara del sujeto.
* **Análisis de iluminación:** estimación de características de iluminación del nuevo escenario, como dirección, color e intensidad.
* **Computer Graphics:** aplicación de un modelo de iluminación para modificar la apariencia de la persona.
* **Composición:** integración de la persona relit con el nuevo escenario.
* **Renderizado:** generación del resultado final como video.

## Tecnologías

* Python
* OpenCV
* NumPy

## Objetivo

Desarrollar un sistema capaz de **adaptar la iluminación de una persona extraída de un video a las condiciones visuales de un escenario diferente**, produciendo una composición visualmente coherente.
