"""
get arguments: path ot the meta graph
load graph into a model
train model
on finish save and send the container back
use paramiko to send the files back over ssh

WE NEED:
training dataset
port
path to the metafile

"""

from Vengine.main import load, train, save
from sys import argv
import zmq
# establishing a zmq connection
assert argv[0] == int
port = argv[0]
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:%s" % port)
socket.send(0)
msg = socket.recv()

# ML model stuff
PATH_TO_META = argv

model = load(PATH_TO_META)