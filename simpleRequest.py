import requests

X_for_predict = [5, 40, 5, 20, 2020, 1, 1, 3, 2, 2]
                 




api_message = requests.post("http://127.0.0.1:5000/api/v1/get_message/", 
                            headers={"Content-Type": "application/json"},
                            json = {"X_for_predict": [X_for_predict]}                           
                           ) 



print(api_message)

if api_message.ok:
    print(api_message.json())


