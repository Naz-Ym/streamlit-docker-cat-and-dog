# pull official base image
FROM  continuumio/miniconda3:latest


# install 
RUN apt-get update && apt-get install libgl1-mesa-glx -y
RUN pip3 install ultralytics==8.0.45 streamlit==1.19.0
# RUN pip3 install torch==1.13.1 torchvision==0.14.1 
# RUN pip3 install streamlit==1.19.0
# RUN pip3 install ultralytics==8.0.45

# copy project
WORKDIR /home/ubuntu/CATS_DOGS
COPY . .
WORKDIR /home/ubuntu/CATS_DOGS/streamlit_app
