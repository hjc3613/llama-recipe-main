#来自于哪个基础镜像
  
FROM nvcr.io/nvidia/pytorch:23.10-py3
 
 
#安装软件包
   
RUN apt update \
    && apt install -y vim wget htop git  language-pack-zh-hans openssh-server sudo \
    && pip install tensor_parallel==1.2.4 omegaconf hydra-core==1.3.2 wandb deepspeed==0.10.0 transformers bitsandbytes trl transformers_stream_generator fire appdirs loralib black py7zr  datasets evaluate accelerate psutil icetk openpyxl tensorboard cpm_kernels loguru pybind11 tiktoken peft -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip uninstall -y transformer-engine
  
#安装软件包
 
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd && apt-get install lrzsz
  
#拷贝pdsh-2.29到容器目录位于/fl-ift/public下面
 
COPY pdsh-2.29 .
  
#源码编译
 
RUN ./configure --with-ssh --with-rsh --with-mrsh --with-mqshell --with-dshgroups && make && make install
 
#COPY flash-attention .
 
#RUN python setup.py install --user
  
#使用pip安装包
RUN pip uninstall -y ninja && pip install ninja && pip install flash-attn==2.5 --no-build-isolation  -i https://pypi.tuna.tsinghua.edu.cn/simple
  
#设置sshd配置文件
 
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config
  
   
# update language settings
# RUN for i in `locale | awk -F'=' '{print $1}'` ; do export  $i="zh_CN.UTF-8"; done
   
# 此处应按照具体情况进行修改：填入使用者host机器的uid、gid、组名、用户名
 
RUN groupadd -f -g 994 docker && groupadd -f -g 2031 hujunchao && groupadd -f -g 2040 med && useradd -m -u 2031 -g hujunchao -G med -s /bin/bash hujunchao \
    && echo 'hujunchao ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers && echo "hujunchao:unisound#atlas..!" | chpasswd
   
# COPY initialize_env.sh /home/hujunchao
   
USER hujunchao
