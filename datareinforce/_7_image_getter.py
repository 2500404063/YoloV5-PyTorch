import pyautogui as pg
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
sock.bind(("0.0.0.0", 7878))
sock.listen(10)

client, remoteIP = sock.accept()
picture_count = 0
while True:
    recv_data = client.recv(100)
    pg.screenshot(f"raw/csgo{picture_count}.jpg")
    picture_count = picture_count + 1