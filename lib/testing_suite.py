class Color:
    def __init__(self, _class):
        self._class = _class

    def initclass(self):
        self._class.__init__(self._class)


class Color2:
    def __init__(self):
        print("Inited")


col = Color(Color2)
col._class.__init__(col._class)
print(col._class)

from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())