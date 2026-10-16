"""Estantes sintéticos por copy-paste (Fase 2).

Módulos:
  objetos     recorta envases del split *train*, les quita el fondo (rembg) y los cachea como PNG RGBA.
  fondos      fondos procedurales (madera, cartulina, gris/negro) y fotos reales sin productos.
  composicion arma UNA imagen de estante de 3 niveles y devuelve sus cajas exactas.
  lote        genera un lote completo (imágenes, etiquetas YOLO 1 clase y multiclase, conteos.csv, data.yaml).
  visualizar  grilla de muestras con cajas y distribución por clase.
"""
