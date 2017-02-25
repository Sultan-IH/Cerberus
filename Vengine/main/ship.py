"""
containerize a model with docker (don't forget to give gpu and )
ship model to different servers

"""
import paramiko as miko
import time
import asyncio
from subprocess import call
from .save import save

# call(["ls", "-l"])
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

COPY {1} home/dev-01/model_training

EXPOSE 4000


"""


def ship(path_to_graph, model, training_file_path, servers: list):
    """
    Convert to tar ball and then push to servers
    :param path_to_graph:
    :param model:
    :param servers: a list of dictionaries
    :return:
    """
    if model:
        """export and then push to servers"""
        save(model.sess, DEFAULT_META_MODEL_NAME)
        path_to_graph = DEFAULT_META_MODEL_NAME

    build_docker_container(path_to_graph)

    for server in servers:
        train_on_server(server,training_file_path)


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
    stdin, stdout, stderr = client.exec_command('docker run'+ MODEL_CONTAINER_NAME+'with output and stuff')
    client.close()
