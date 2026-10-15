# DPD Cadence Pipeline

Version de trabajo: modelos MP, EMP, GMP, Pruned Volterra y LUT; comparacion IQ; puente de inferencia y exportacion PWL para Cadence.

- framework_dpd/: entrenamiento, modelos guardados y cadence_bridge.
- procesador_iq_pa/: preprocesamiento IQ. El netlist de terceros y los datos brutos de este procesador no se distribuyen.
- comparador_constelaciones_5_modelos/: comparador independiente.
- framework_dpd/data/cadence_exports/run_20261001T174609_515799Z/: estimulos, incluido Volterra_limite_3V_ceros.

Se conserva la version actual del codigo. No se incluyen entornos virtuales, PDK, netlists ni librerias de Cadence. Algunas rutas historicas de procedencia se conservan en metadatos; adaptar las rutas de ejecucion al clonar.

Documentacion: framework_dpd/cadence_bridge/README.md. Las simulaciones Spectre y la validacion del servidor Python 3.6 siguen pendientes.
