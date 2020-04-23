# FaceNet MTCNN 在线实时人脸检测

## 模型管理

每个模型训练后打包的pb模型文件，统一管理:

新建模型目录

```shell
mkdir /data/mutil_model/
```

各个模型管理目录tree

```shell
├── faster_rcnn
│   └── 1
│       ├── saved_model.pb
│       └── variables
│           ├── variables.data-00000-of-00001
│           └── variables.index
├── models.config
└── mtcnn
    ├── 1
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    ├── 2
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    ├── 3
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    └── README.txt
```

models.config文件

```shell
model_config_list:{
    config:{
        name:"faster_model",
        base_path:"/models/mutil_model/faster_rcnn",
        model_platform:"tensorflow"
    },
    config:{
        name:"mtcnn_model",
        base_path:"/models/mutil_model/mtcnn",
        model_platform:"tensorflow",
        model_version_policy:{
            all:{}
        }
    },
}
```

模型存在多个版本的时：

```shell
    model_version_policy:{
        all:{}
      }
```

## Docker启动模型

```shell
# 方式一:
nvidia-docker run -d --rm -it --name=facenet \
        --network=mynetwork \
        --ip=172.20.0.18 \
        --mount type=bind,source=/data/mutil_model/saved_model/,target=/models/saved_model \
        -e CUDA_VISIBLE_DEVICES=0 \
        --entrypoint=tensorflow_model_server tensorflow/serving:1.11.0 \
        --port=8500 --per_process_gpu_memory_fraction=0.2 \
        --rest_api_port=8501 --model_name=facenet_pb --model_base_path=/models/facenet_pb

# 方式二使用 models.config
nvidia-docker run -p 8501:8501 \
    --name container_name \
    --mount type=bind,source=/data/mutil_model/mutil_model/,target=/models/mutil_model \
    -t tensorflow/serving --model_config_file=/models/mutil_model/models.config

# 其他方式
docker run -t --rm -p 8501:8501 \
    --name container_name \
    -v "/data/mutil_model/saved_model/:/models/saved_model" \
    -e MODEL_NAME=saved_model \
    tensorflow/serving
```

请求ip

```shell
#查看状态：
curl http://localhost:8501/v1/models/faster_model

{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}

curl http://localhost:8501/v1/models/mtcnn_model

metadata

curl http://localhost:8501/v1/models/mtcnn_model/metadata

# 请求url
"http://localhost:8501/v1/models/mtcnn/1/mtcnn_model:predict"
"http://localhost:8501/v1/models/faster_rcnn/faster_model:predict"
```
