import zmq
import random
import time
import http.server
import socketserver
from http.server import BaseHTTPRequestHandler,HTTPServer
from threading import Thread
import serial
port = "6000"
topic = "image"
topic_2 = "presence"
PORT = 8080
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % port)


def deliver_data():
    while True:
        for i in range(0,5):
            read_string = ser.readline().decode('utf-8').replace("b'", '')[:-4].replace('.','').split(',')
            if len(read_string) < 5:
                raw_data = np.append(raw_data,read_string)
        print(raw_data)
        data = {"cylinder":"V8_pi","temp_average":1,"photo_average":1,"gas_average":1}
        data["temp_average"] = raw_data[0]
        data["photo_average"] = raw_data[2]
        data["gas_average"] = raw_data[4]
        r = requests.post('https://inventor-s-hub.xyz/v8', data=data)
        time.sleep(300)
        

class Handler(BaseHTTPRequestHandler):

    # Handler for the GET requests
    def do_GET(self):
        print('Get request received')
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        # Send the html message
        print(self.path)
        if self.path == "/v8":
            self.wfile.write(bytearray("Hello you have navigated to v8!","utf-8"))
            socket.send_string("{0} {1}".format(topic_2, "present"))
            print(topic)
        else:
            self.wfile.write(bytearray("Hello world","utf-8"))
        return


httpd = socketserver.TCPServer(("", PORT), Handler)

print("serving at port", PORT)

t = Thread(target=httpd.serve_forever())
t.start()
t2 = Thread(target=deliver_data())
t2.start()
