import socket
import json
import numpy as np


def receive_all(sock, buffer_size=40960):
    """接收所有数据直到没有更多为止"""
    data = b''
    cnt = 0
    while True:
        cnt += 1
        # print('recieve times=%d' % cnt)
        part = sock.recv(buffer_size)
        data += part
        if len(part) < buffer_size:
            # 没有更多数据可读或缓冲区未满
            break
    return data


def send_data(client_socket, data, max_attempts=5):
    """
    向客户端发送数据，并在失败时重试。

    :param client_socket: 客户端套接字。
    :param data: 需要发送的数据（字典格式）。
    :param max_attempts: 最大尝试次数。
    :return: 发送成功返回 True，所有尝试失败返回 False。
    """
    # 将字典数据转换为 JSON 格式的字符串并编码为字节
    encoded_data = json.dumps(data).encode('utf-8')

    # 尝试发送数据
    for attempt in range(max_attempts):
        try:
            client_socket.sendall(encoded_data)
            return True  # 数据发送成功
        except Exception as e:
            print(f"Send attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                return False  # 所有尝试都失败


def start_env_server(host='127.0.0.1', port=12345):
    # 创建 socket 对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定端口
    server_socket.bind((host, port))

    # 开始监听
    server_socket.listen(5)
    print(f"Listening on {host}:{port}")

    return server_socket


def start_server(host='127.0.0.1', port=12345):
    # 创建 socket 对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定端口
    server_socket.bind((host, port))

    # 开始监听
    server_socket.listen(5)
    print(f"Listening on {host}:{port}")

    while True:
        # 等待客户端连接
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        while True:
            # 接收数据
            data = receive_all(client_socket)
            if not data:
                break  # 断开连接
            states_rewards_dones = data.decode('utf-8')
            # print(states_rewards_dones)
            # response_data = {'state': [1, 2, 3]}  # 示例状态数据
            # client_socket.sendall(json.dumps(response_data).encode('utf-8'))

            # 解析接收到的数据
            try:
                json_data = json.loads(states_rewards_dones)
                print(json_data)

                # 发送回应数据
                num_agents = 50
                actions = []
                for _ in range(num_agents):
                    actions.append({'floats': [np.random.random(), np.random.random(), np.random.random()*0.25*np.pi]})
                actions_data = {'actions': actions}
                # response_data = {'state': [1, 2, 3]}  # 示例状态数据
                client_socket.sendall(json.dumps(actions_data).encode('utf-8'))
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                break

        # 关闭当前客户端连接
        client_socket.close()
        print(f"Connection closed from {addr}")

    # 关闭服务器 socket
    server_socket.close()


if __name__ == '__main__':
    start_server()
