# 如果容器存在，则删除
if docker ps -a --format '{{.Names}}' | grep -q "^pytorch$"; then
    echo "Removing existing container 'pytorch'..."
    docker rm -f pytorch
fi

docker run --name pytorch --gpus all -it \
        -v ./:/workspace \
        --shm-size 8g \
        my_pytorch:v2 \
        bash