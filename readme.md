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
- Prueba `reward_clip: true` si observas inestabilidad en el aprendizaje.
- Mantén trazabilidad: cada run guarda su `config.json` y su `logs/*.log`.

Siguientes pasos (opcionales que puedo añadir):
- Añadir integración con Weights & Biases o MLflow para tracking automático.
- Scripts de búsqueda de hiperparámetros (sweep) sobre el YAML.
- Tests unitarios básicos para wrappers/envs.

Si quieres, genero un `configs/` con ejemplos listos y un `Makefile`/scripts para lanzar experimentos en Windows (PowerShell).

