# RL CarRacing — Proyecto pulido y listo para experimentar

Este repositorio entrena y evalúa un agente PPO en el entorno CarRacing (Gymnasium) usando entradas visuales (píxeles). He mejorado la estructura para que puedas
centrarte únicamente en los hiperparámetros: ahora puedes cargar un archivo YAML/JSON con todos los parámetros o ajustar unos pocos flags desde la línea de comandos.

Principales mejoras realizadas:
- Configuración centralizada y cargable desde YAML/JSON (`src/config.py`).
- Guardado automático de la configuración por ejecución (`results/<run>/config.json`).
- Logging a consola y fichero para cada run (`src/utils.configure_logging`).
- Selección automática/forzada de dispositivo (CPU/CUDA).
- Scripts `train.py` y `eval.py` con argumentos CLI y trazabilidad.

Requisitos mínimos
- Python 3.9+
- Instala dependencias:

```powershell
pip install -r requirements.txt
```

Uso rápido

1) Entrenar con la configuración por defecto:

```powershell
python -m src.train
```

2) Entrenar usando un archivo de configuración YAML (ejemplo abajo):

```powershell
python -m src.train --config configs/ppo_carracing.yaml
```

3) Sobrescribir solo algunos parámetros desde la CLI (ej.: número de entornos o timesteps):

```powershell
python -m src.train --n-envs 8 --total-timesteps 1000000
```

4) Evaluar y grabar vídeos de un modelo entrenado:

```powershell
python -m src.eval --model results/<RUN_ID>/models/best_model.zip
```

Formato de configuración (YAML) — ejemplo mínimo `configs/ppo_carracing.yaml`:

```yaml
env_id: CarRacing-v3
seed: 42
frames_stack: 4
grayscale: true
resize: 84
frame_skip: 2
total_timesteps: 2000000
learning_rate: 2.5e-4
n_steps: 2048
batch_size: 256
n_epochs: 10
run_name: ppo_carracing
log_root: results
eval_freq: 50000
n_eval_episodes: 10
save_freq: 100000
```

Modo visual (opcional): ver entorno original y lo que ve el modelo
---------------------------------------------------------------

Si quieres ver en tiempo real lo que ocurre durante el entrenamiento puedes usar el flag `--visual`.
Por defecto el entrenamiento se ejecuta sin visualización para maximizar velocidad. `--visual` está pensado
para debugging y experimentos cortos, no para correr entrenamientos de largo tiempo.

Qué muestra `--visual`:
- Una ventana llamada `env_raw` con el render RGB original del entorno (imagen completa con interfaz del entorno).
- Otra ventana llamada `env_preprocessed` con la versión que se pasa al modelo (grayscale o color según la configuración,
	redimensionada a la resolución configurada en `frames_stack`/`resize`).

Comportamiento técnico:
- Al activar `--visual` el runner fuerza `n_envs=1` y crea los entornos con `render_mode='rgb_array'` para poder
	capturar frames como arrays. Si intentas usar `--visual` con múltiples entornos la opción será silenciosamente forzada a 1.
- La visualización usa OpenCV (`cv2.imshow`) y actualiza las ventanas cada paso del agente. Esto puede ralentizar
	significativamente el entrenamiento; usa `--visual` para inspección, no para runs largos.
- Si el sistema no dispone de interfaz gráfica (servidor sin X/Wayland, WSL sin servidor X), la visualización no funcionará.

Cómo usarlo (ejemplos):

Entrenar y ver la visualización en tiempo real:

```powershell
python -m src.train --visual
```

Entrenar con config personalizada y visualización:

```powershell
python -m src.train --config configs/ppo_carracing.yaml --visual
```

Controlar la frecuencia de renderizado
------------------------------------

Si la visualización ralentiza demasiado el entrenamiento puedes reducir la frecuencia de actualización
con `--render-freq`. Por ejemplo, `--render-freq 5` actualizará las ventanas cada 5 pasos del agente (reduce I/O
y CPU usados por OpenCV):

```powershell
python -m src.train --visual --render-freq 5
```

Opciones adicionales para reducir overhead
----------------------------------------

Si necesitas más eficiencia en modo visual, hay dos flags útiles:

- `--visual-only-preproc`: muestra únicamente la imagen preprocesada que recibe el modelo (desactiva `env_raw`). Esto reduce CPU/IO al evitar convertir y mostrar el render completo.
- `--visual-scale N`: escala la imagen preprocesada por un factor N para verla mejor. Usa valores pequeños (p.ej. 2 o 4). Por ejemplo, `--visual-scale 4` mostrará 84x84 como ~336x336.

Ejemplo combinando opciones para menos impacto:

```powershell
python -m src.train --visual --render-freq 10 --visual-only-preproc --visual-scale 3
```


Archivo de configuración de ejemplo
----------------------------------

He añadido un ejemplo listo para usar en `configs/ppo_carracing.yaml`. Puedes copiarlo y adaptarlo.
Ejecuta con:

```powershell
python -m src.train --config configs/ppo_carracing.yaml
```

Notas y recomendaciones:
- Si ves que la visualización frena demasiado el entrenamiento, para inspección reduce el número de pasos
	o usa `--visual` solo durante los primeros minutos.
- La visualización está diseñada para mostrar tanto el entorno original como la entrada preprocesada del modelo,
	lo que facilita detectar problemas en el pipeline de observaciones (por ejemplo, problemas de normalización, cambio
	de canales, o errores en el resizing).

¿Qué hace cada script?
- `python -m src.train`: carga la configuración (o usa la por defecto), crea directorios en `results/<run_name>/<timestamp>/`, configura logging y entornos, y ejecuta `model.learn`. Guarda checkpoints, el mejor modelo y un `final_model.zip`.
- `python -m src.eval --model <ruta>`: carga el modelo y ejecuta episodios de evaluación (guarda vídeos si es necesario). Reporta reward medio y desviación estándar.

Buenas prácticas y recomendaciones
- Empieza con `n_envs=1` para probar que todo funciona; si tienes CPU y memoria, sube a 4 o 8 para velocidad.
 - Empieza con `n_envs=1` para probar que todo funciona; si tienes CPU y memoria, sube a 4, 8 o más para velocidad.
 - Usa `SubprocVecEnv` (se activa automáticamente cuando `n_envs>1`) para paralelizar las llamadas a `env.step` entre procesos.
 - Ajusta el número de hilos de cálculo con `--num-threads` si ves sobrecarga de CPU (p. ej. `--num-threads 8`).
 - Si tienes GPU, fuerza `--device cuda` para mover el entrenamiento a la GPU (aumenta throughput si la red y optimización son el cuello de botella).
 
Por qué ejecutar con `--n-envs 8 --num-threads 8` aumenta la eficiencia
-----------------------------------------------------------------

Comando recomendado para rendimiento:

```powershell
python -m src.train --config configs/ppo_carracing.yaml --n-envs 8 --num-threads 8
```

Explicación práctica y técnica:

- Paralelismo de entornos (SubprocVecEnv / `--n-envs`):
	- Al usar `n_envs>1` el código crea múltiples entornos que corren en procesos separados (SubprocVecEnv). Eso paraleliza las llamadas a `env.step()` y al preprocesado por frame (resize, gris, etc.).
	- En vez de ejecutar paso-a-paso un único entorno en el proceso principal (coste Python + llamadas a C por cada frame), se hacen múltiples pasos simultáneos en procesos distintos, lo que suele multiplicar los samples/segundo.
	- PPO puede aprovechar esto porque toma muestras en paralelo y acumula `n_steps * n_envs` experiencias antes de hacer un update; así reducimos overhead por interacción y aprovechamos mejor la CPU/GPU.

- Control de hilos (BLAS / PyTorch `--num-threads`):
	- Bibliotecas numéricas (OpenBLAS, MKL, PyTorch) crean threads internos para operaciones lineales. Si dejas que cada proceso cree muchos threads y además lanzas `n_envs` procesos, terminas con demasiados hilos activos y alto overhead de scheduling (context switching).
	- `--num-threads` fija el número de threads que usa PyTorch/BLAS por proceso (mediante `torch.set_num_threads`). Un valor razonable (p. ej. 8 en una CPU de 8 núcleos) evita sobre-subscription y mejora la utilización real de los núcleos.

- Relación con batch size y latencia de actualización:
	- Con `n_envs=8`, si `n_steps=2048`, la actualización ve `2048*8` pasos por update — mayúsculo batch que amortigua el coste del optimizador.
	- Si subes `n_envs` conviene revisar `n_steps` y `batch_size` para mantener tamaños de batch apropiados y no subir la latencia de las actualizaciones más de lo deseado.

- Trade-offs y recomendaciones:
	- Más `n_envs` = más throughput (it/s) pero mayor uso de memoria y CPU. Empieza por 4, luego 8, y monitoriza uso de CPU/RAM.
	- Ajusta `--num-threads` para que `n_envs * num_threads` no exceda el número de hardware threads disponibles (p. ej. 8 envs × 8 threads = 64 hilos — esto puede ser excesivo en máquinas pequeñas).
	- Si la GPU es el cuello (modelo grande), usa `--device cuda`; en ese caso la mejora por aumentar `n_envs` suele ser aún más significativa porque la GPU procesa batches más grandes eficientemente.

Medición y ajustes
------------------

Para encontrar la combinación óptima en tu máquina, itera: prueba `n_envs` = 2,4,8 y `--num-threads` = 1,2,4,8. Mide:

- it/s (iteraciones por segundo) reportadas por SB3
- uso de CPU y porcentaje por núcleo (Task Manager / htop)
- uso de memoria y GPU (si aplica)

Con esos datos elige el punto donde `it/s` se acerca al máximo antes de que CPU/GPU se sature o la memoria se agote.

Si quieres, puedo añadir un pequeño script `bench_envs.py` que automatice estas pruebas en tu máquina y te devuelva la mejor configuración recomendada.
- Prueba `reward_clip: true` si observas inestabilidad en el aprendizaje.
- Mantén trazabilidad: cada run guarda su `config.json` y su `logs/*.log`.

Visualizar curvas con TensorBoard
--------------------------------

Para seguir el progreso del entrenamiento y ver las curvas de aprendizaje (pérdida, entropía, rewards y las métricas de evaluación) usamos TensorBoard. El entrenamiento ya está configurado para exportar logs a la carpeta `results` (cada run crea `results/<run_name>/<timestamp>/logs`).

Instrucciones rápidas:

- Instalar TensorBoard si no lo tienes:

```powershell
python -m pip install tensorboard
```

- Lanzar TensorBoard (Windows / PowerShell, local):

```powershell
tensorboard --logdir results --port 6006
# Abrir en el navegador: http://localhost:6006
```

- En Colab (celda Python):

```python
# en Colab monta primero Drive si los logs están ahí
%load_ext tensorboard
%tensorboard --logdir /content/proyecto_rl/results
```

Notas sobre qué verás:
- `eval/mean_reward`, `eval/std_reward`, `eval/mean_length`, `eval/std_length` (métricas de evaluación que añadimos). Estas aparecen cada vez que corre la evaluación periódica (`eval_freq`).
- Métricas internas de entrenamiento grabadas por Stable-Baselines3 (p. ej. loss/actor, loss/critic, entropy, learning_rate, etc.).
- Para comparar runs: apunta TensorBoard al directorio `results` raiz; verás cada `run_name/timestamp` como experimento separado.

Consejos prácticos:
- Si no ves las métricas de evaluación, verifica que el callback de evaluación está activo y que `eval_freq` y `n_eval_episodes` están correctamente definidos en tu config/CLI.
- Aumenta `n_eval_episodes` si la varianza es alta para obtener estimaciones más estables en TensorBoard.
- Guarda checkpoints frecuentes si planeas analizar curvas a partir de distintos puntos de entrenamiento.

Siguientes pasos (opcionales que puedo añadir):
- Añadir integración con Weights & Biases o MLflow para tracking automático.
- Scripts de búsqueda de hiperparámetros (sweep) sobre el YAML.
- Tests unitarios básicos para wrappers/envs.

Si quieres, genero un `configs/` con ejemplos listos y un `Makefile`/scripts para lanzar experimentos en Windows (PowerShell).

