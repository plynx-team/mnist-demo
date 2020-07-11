FROM tensorflow/tensorflow:2.3.0rc1

WORKDIR /app

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY ./main.py /app/main.py
COPY ./mnist_demo /app/mnist_demo

CMD python main.py
