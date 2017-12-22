import paramiko as miko
import time
import zmq
import os
from subprocess import call
from .save import save
import socket

LOCAL_IP = socket.gethostbyname(socket.gethostname())
if LOCAL_IP == "127.0.0.1" or LOCAL_IP is None:
    LOCAL_IP = socket.gethostbyname(socket.getfqdn())

TRAIN_SCRIPT_PATH = "./train_model_script.py"  # maybe change this to os.path() or smt
CURRENT_TIME = time.strftime("%d/%m/%Y")

DOCKERFILE = """

FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y --force-yes python3.5
RUN apt-get install -y --norecoomendatoions python-pip
RUN pip install Cerberus

RUN mkdir home/{0}

COPY {1} home/{0} # copy tarball

EXPOSE 4000


"""


def ship(model, servers: list):
    model_dir = "./" + model.name
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if model != str:
        save(model.sess, model_dir)

    build_docker_container(model_dir, model.name)

    for server in servers:
        train_on_server(server, model.name)


"""Helper methods"""


def build_docker_container(dir_path, name):
    call("docker build -t " + name + " . ")
    call("docker save " + name + " > " + dir_path + name + ".tar")
    with open("./Dockerfile", "w") as f:
        f.write(DOCKERFILE.format(dir_path, name + ".tar"))


def train_on_server(server, name):
    client = miko.SSHClient()
    client.connect(
        hostname=server["url"],
        username=server["username"],
        password=server["password"])
    # copy the tar ball
    # init training and catch response
    client.exec_command("docker load < " + name)
    client.exec_command("docker run -it " + name + "/bin/python3 /home/"+name+"/train_model_script.py")
    client.exec_command("")

    client.close()


def create_pair():
    port = "5556"
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://localhost:%s" % port)

    while True:
        msg = socket.recv()
        socket.send("client message to server1")
        socket.send("client message to server2")
        time.sleep(1)
