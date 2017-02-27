"""
In the server:
docker load < form tar ball
docker-nvidia run model_container python3.5 train_model_script.py
ship container back
IS THERE A POINT IN DOCKER CONTAINERS?

TODO: establish two way communication
PROBLEM: how do you send a meta file back to the hosts machine? Do you use scp if the user is logged on

"""
import paramiko as miko
import time
import zmq
from subprocess import call
from .save import save
import socket

LOCAL_IP = socket.gethostbyname(socket.gethostname())
if LOCAL_IP == "127.0.0.1" or LOCAL_IP is None:
    LOCAL_IP = socket.gethostbyname(socket.getfqdn())

TRAIN_SCRIPT_PATH = "./train_model_script.py"  # maybe change this to os.path() or smt
CURRENT_TIME = time.strftime("%d/%m/%Y")
TAR_BALL_NAME = 'model_container' + CURRENT_TIME + '.tar'
DEFAULT_META_MODEL_NAME = 'model' + CURRENT_TIME + '.meta'
MODEL_CONTAINER_NAME = 'model_container_' + CURRENT_TIME

DOCKERFILE = """

FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y --force-yes python3.5
RUN apt-get install -y --norecoomendatoions python-pip

RUN mkdir home/model_training

COPY {1} home/model_training # copy tarball
COPY {2} home/model_training

EXPOSE 4000


"""


def ship(model, docker: bool, servers: list):
    """
    Convert to tar ball and then push to servers
    :param model:
    :param docker:
    :param servers: a list of dictionaries
    :return:
    """
    if model != str:
        """export and then push to servers"""
        save(model.sess, DEFAULT_META_MODEL_NAME)
        model = DEFAULT_META_MODEL_NAME

    build_docker_container(model)
    if docker:
        pass
    for server in servers:
        train_on_server(server, training_file_path)


"""Helper methods"""


def build_docker_container(export_path):
    with open("dockerfile", "w") as f:
        f.write(DOCKERFILE.format(export_path))
    call("docker build -t " + MODEL_CONTAINER_NAME + " . ")  # TODO: SHOULD HAVE A KIND OF NAMING SYSYTEM
    call("docker save _model > " + TAR_BALL_NAME)


def train_on_server(server, path_to_train_file):
    client = miko.SSHClient()
    client.connect(
        hostname=server["url"],
        username=server["username"],
        password=server["password"])
    call("scp " + path_to_train_file + " username@a:/path/to/destination")
    stdin, stdout, stderr = client.exec_command('docker run' + MODEL_CONTAINER_NAME + 'with output and stuff')
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
