import iperf3


def bandwidthServer():
    server = iperf3.Server()
    print("带宽测量启动")
    while True:
        result = server.run()
        print("measure band width from:" + result.remote_host)

if __name__ == "__main__":
    bandwidthServer()