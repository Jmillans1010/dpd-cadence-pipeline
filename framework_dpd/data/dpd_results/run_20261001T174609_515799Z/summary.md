# Identificación inversa DPD

Test: segunda mitad de Validation, excluida de selección de hiperparámetros.

| Modelo | NMSE train dB | NMSE val dB | NMSE test dB | BIC dB / progreso histórico | Complejidad | Parada |
|---|---:|---:|---:|---:|---:|---|
| MP | -30.4192 | -28.8252 | -29.8447 | -30.1569 | 54 | search_limit_reached |
| EMP | -23.6362 | -23.4246 | -23.4793 | -23.4127 | 46 | search_limit_reached |
| GMP | -38.4093 | -31.7582 | -29.0874 | -37.4472 | 198 | bic_minimum_reached |
| PrunedVolterra | -39.0257 | -30.9262 | 28.6423 | -36.9751 | 422 | numerical_instability |
| LUT | -22.4306 | -22.2919 | -22.5275 | -22.1196 | 68 | numerical_frontier_incomplete |

Son errores de identificación del inverso, no EVM/ACLR del PA linealizado.
BIC usa error de entrenamiento y penaliza coeficientes complejos según Becerra et al. (2020). Menor es mejor.
Una parada local por BIC no garantiza un óptimo global. Los límites e inestabilidad se informan por separado.

- Los tiempos de simulaciones separadas no establecen cronología global.
- La independencia estadística de señales PN9 no se demuestra con hashes distintos.

Observaciones de esta ejecución:

- MP y EMP agotaron los límites ampliados: todavía no hay meseta BIC demostrada.
- GMP encontró un mínimo BIC local, con mayor error de test que en la ejecución previa; no equivale a óptimo de generalización.
- Volterra presenta NMSE de test +28.64 dB y no es apto como modelo validado. Su búsqueda terminó por inestabilidad numérica.
- LUT dejó vecinos sin evaluar por estabilidad numérica: la frontera BIC está incompleta.
- La ampliación del contexto cambia ligeramente las muestras puntuadas respecto de las ejecuciones previas.
