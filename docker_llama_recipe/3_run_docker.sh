

docker run --name $1_v0  -itd  --gpus all  -p 64002:34002 -p 64003:34003  -p 64004:34004  -p 64005:34005 -p 64022:22 -v /fl-ift/med/:/fl-ift/med/    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864   harbor.unijn.cn/hujunchao/$1:v0
