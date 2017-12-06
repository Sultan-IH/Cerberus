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

# if a two way connection is established then
two modes? one interactive and one autonomous

"""

from Vengine.main import load, train, save
from sys import argv

# ML model stuff
PATH_TO_META = argv


model = load(PATH_TO_META)