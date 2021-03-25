import socket
import threading
import pickle
import iperf3
import ctypes
import time

IP="127.0.0.1"
PORT=8081

def sendData(server, inputData):
	server.sendall(inputData)

def receiveData(server):
    conn, addr = server.accept()
    print('Connected by', addr)

    data = conn.recv(1024)
    print("data:" + str(data))

    sendData(conn, b'Thanks')


def BandWidthServer():
    server = iperf3.Server()
    while True:
        result = server.run()
        print("measure band width from:" + result.remote_host)

if __name__ == "__main__":
    t = threading.Thread(target=BandWidthServer, name='BandWidthServer')
    t.daemon = True
    t.start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.bind((IP, PORT))
	print("云端启动，准备接受任务")
	server.listen()
	receiveData(server)