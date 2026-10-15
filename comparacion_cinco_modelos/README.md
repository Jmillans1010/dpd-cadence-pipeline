# Comparación: salida medida a entrada reconstruida

Se evalúan directamente los cinco modelos de run_20261001T174609_515799Z,
sin inversión numérica ni reentrenamiento. Entrada del modelo: z=y/G, usando la ganancia
compleja del manifiesto original. Referencia: entrada medida x. Predicción: x_estimada=F(z).
Esto evalúa el modelo inverso como postdistorsionador; no demuestra por sí mismo
la linealización obtenida al aplicar un predistorsionador antes del amplificador.

Se conserva la alineación original. Todos usan los mismos 4336 índices de test y
4347 índices de validación, guardas y contexto del manifiesto (incluye adelantos GMP).
Los diez NMSE reproducen los valores guardados durante el entrenamiento con tolerancia 1e-8 dB.
Criterio de selección: bic.
En BIC, validación es diagnóstica; en reference_fraction participa en selección.
Test es la segunda mitad reservada del registro Validation.

| Modelo | NMSE test dB | Error RMS relativo % | RMSE | NMSE validación dB |
|---|---:|---:|---:|---:|
| MP | -29.844728 | 3.219316 | 0.00727276 | -28.825213 |
| EMP | -23.479340 | 6.699355 | 0.0151345 | -23.424577 |
| GMP | -29.087446 | 3.512592 | 0.0079353 | -31.758214 |
| PrunedVolterra | 28.642263 | 2704.662820 | 6.1101 | -30.926159 |
| LUT | -22.527509 | 7.475229 | 0.0168873 | -22.291901 |

NMSE=10 log10(sum(|x-x_estimada|²)/sum(|x|²)). Error RMS relativo=100 sqrt(error
normalizado); no es EVM contra símbolos ideales. RMSE en unidades del CSV.
Los gráficos usan escala común salvo paneles con predicciones extremas, que se
muestran completos con escala ampliada indicada en el título. Usan todas las muestras temporales,
incluidas transiciones; no hay nuevo ajuste de ganancia, fase, retardo o instante de símbolo.

Los resultados anteriores de inversión de MP, LUT y Volterra se preservan en
historico_inversion/ dentro de cada workspace antiguo. Sus comparar.py, README y
resultados actuales ahora corresponden a este ejercicio corregido.

Ejecutar: python comparar.py (numpy y matplotlib). Por defecto usa latest_run.json.
Para otra ejecución: python comparar.py --run run_<fecha>. Las copias en input/ y modelos/
se actualizan juntas desde esa ejecución; el CSV se verifica por SHA-256.
Cada modelo tiene CSV separados de validación y test, métricas y constelación en resultados/.
