## 从天池基础镜像构建(from的base img 根据自己的需要更换，建议使用天池open list镜像链接：https://tianchi.aliyun.com/forum/postDetail?postId=67720)
# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.6-cuda10.1-py3
FROM registry.cn-shanghai.aliyuncs.com/eboxliu/tianchi2021:package

# 国内镜像源
# RUN mirror_url=mirrors.tuna.tsinghua.edu.cn && \
#     sed -i "s/security.ubuntu.com/$mirror_url/" /etc/apt/sources.list && \
#     sed -i "s/archive.ubuntu.com/$mirror_url/" /etc/apt/sources.list && \
#     sed -i "s/security-cdn.ubuntu.com/$mirror_url/" /etc/apt/sources.list && \
#     pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
#     apt update && \
#     apt -y install vim

# RUN mirror_url=mirrors.aliyun.com && \
#     sed -i "s/security.ubuntu.com/$mirror_url/" /etc/apt/sources.list && \
#     sed -i "s/archive.ubuntu.com/$mirror_url/" /etc/apt/sources.list && \
#     sed -i "s/security-cdn.ubuntu.com/$mirror_url/" /etc/apt/sources.list && \
#     pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
#     apt update && \
#     apt -y install vim

## 把当前文件夹里的文件构建到镜像的工作目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR  /

##安装依赖包,pip包请在requirements.txt添加
# RUN pip install --no-cache-dir ./segmentation_models.pytorch && \
#     pip install --no-cache-dir -r requirements.txt && \
#     rm -rf segmentation_models.pytorch && \
#     rm requirements.txt

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]