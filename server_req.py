from ai_hub import inferServer
from flask import Flask, request
import requests
import json
import base64
import time


def loader2json(data):
    # send_json = json.loads(base_json, encoding='utf-8')
    send_json = {}
    bast64_data = base64.b64encode(data)
    bast64_str = str(bast64_data,'utf-8')
    send_json['img'] = bast64_str
    send_json = json.dumps(send_json)
    # print('send_json:', send_json)
    return send_json

def send_eval(data):
    url = "http://127.0.0.1:8080/tccapi"
    start = time.time()
    res = requests.post(url, data, timeout=3)
    cost_time = time.time() - start
    # res = analysis_res(res)
    return res, cost_time

fin=open("data_sets/val/000184.tif",'rb')
img=fin.read()
data_json = loader2json(img)
ret, cost_time = send_eval(data_json)
print("ret:", ret, "cost_time:", cost_time)