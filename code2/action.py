import requests
import time
ESP32_IP = "172.16.204.110" #IP地址可能会发生变动！

ACTIONS = {"up","down","left","right","stop","grab","release"}

#记录上一次发送的动作和时间
_LAST = {"act":None,"ts":0.0}
_MIN_INTERVAL = 0.15 #最小发送时间


def send_control_command(action,speed = 50,times=None,force=False):
    #向ESP32发送控制命令 (去抖动)
    global _LAST

    if action not in ACTIONS:
        print(f"error{action}")
        return
    
    now = time.time()

    if(not force)and action == _LAST["act"]and (now - _LAST["ts"]) < _MIN_INTERVAL:
        return 

    params = {
        "action":action,
        "speed":speed
    }

    if times is not None:
        params["times"] = times

    try:
        url = f"http://{ESP32_IP}/control"
        response = requests.get(url,params=params,timeout=1)
        print(f"finish send action {response.url}")
    except Exception as e:
        print(f"error:{e}")

    _LAST["act"] = action
    _LAST["ts"] = now



if __name__  == "__main__":
    send_control_command("up", time=10000)