# Comparador independiente de cinco modelos

Abrir resultados/comparador.html: panel interactivo autónomo, funciona sin internet.
Ejecutar ejecutar.ps1 o `python comparar.py` para regenerarlo.
Dependencias: `python -m pip install -r requirements.txt`.

Se incluyen copias de los cinco modelos, el manifiesto y el CSV de run_20261001T174609_515799Z.
No se consulta ni modifica ningún otro workspace al ejecutar el comparador.
Entrada del modelo: salida medida y dividida por la ganancia compleja G original.
Predicción: entrada reconstruida x_estimada. Referencia: entrada medida x.
I y Q se expresan en voltios; RMSE también en voltios. No se normalizan las
constelaciones gráficas a potencia unitaria. No hay nuevos ajustes de fase o ganancia.

Test: 4336 muestras. Se verifican los diez NMSE de validación/test frente
a los resultados originales, con tolerancia 1e-8 dB. La validación participó en
selección de modelos; test corresponde al segmento reservado del mismo registro.
Se incluyen todas las muestras temporales y transiciones, no solo centros de símbolo.

La vista completa mantiene la misma escala para todos y muestra todos los puntos.
El zoom alrededor de la entrada puede ocultar extremos, identificados en la figura PNG.
Los resultados de Pruned Volterra en esta ejecución tienen errores muy grandes;
se preservan íntegramente. La escala completa permite examinarlos.

Archivos: resultados/comparador.html, constelaciones_completa.png/svg,
constelaciones_zoom_entrada.png/svg, metricas.json y CSV por modelo/partición.
NMSE=10log10(sum(|x-x_estimada|²)/sum(|x|²)). RMS relativo=100sqrt(error normalizado).
No es EVM contra símbolos ideales ni prueba de linealización con DPD antes del PA.
