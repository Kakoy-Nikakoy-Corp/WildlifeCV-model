## Установка

```bash
git clone https://github.com/Kakoy-Nikakoy-Corp/WildlifeCV-model.git
cd WildlifeCV-model

uv sync
```

## Использование
> Чтобы эти команды работали, вы должны создать корректный `.env` файл внутри корневой директории. См. шаблон ниже.

Запуск обучения модели (с логированием в MLFlow):
```bash
uv run train
```

Валидация на тестовой выборке (с логированием в MLFlow):
```bash
uv run validate
```

Запуск `mlflow ui` для отслеживания результатов обучения:
```bash
uv run serve-ui
```

Тюнинг гиперпараметров:
```bash
uv run tune
```

Экспорт модели в TensorRT-формат:
```bash
uv run export
```

## .env шаблон
Предполагается, что мы подключены к `192.168.0.104` посредством WireGuard VPN, 
и PostgreSQL+Garage развернуты на соответствующих портах. В PostgreSQL создан пользователь `irbis` и принадлежащая ему база данных `mlruns`.
```env
# YOLO
RUN=run_gpu_5
MODEL_PATH=models/yolo26x.pt
DATASET_URL=https://dataset.irbis.wild1.net/dataset.ndjson

# MLFlow
EXPERIMENT_NAME=WildlifeCV

# PostgreSQL integration
MLFLOW_TRACKING_URI=postgresql://irbis:<PASSWORD>@192.168.0.104:5432/mlruns

# Garage S3 integration
ARTIFACT_URI=s3://irbis
MLFLOW_S3_ENDPOINT_URL=http://192.168.0.104:3900
AWS_ACCESS_KEY_ID=<16 DIGIT HEX KEY>
AWS_SECRET_ACCESS_KEY=<32 DIGIT HEX KEY>
AWS_DEFAULT_REGION=garage
```
