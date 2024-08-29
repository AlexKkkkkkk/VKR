from flask import (Flask, request, jsonify, render_template)

import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


region_of_moscow_LE = pickle.load(open("/Users/krop/Desktop/Data science/ВКР/prod/flaskApp/models/region_of_moscow_LE.pkl","rb",))

num_scaler = pickle.load(open("/Users/krop/Desktop/Data science/ВКР/prod/flaskApp/models/num_scaler.pkl", "rb"))

model = pickle.load(open("/Users/krop/Desktop/Data science/ВКР/prod/flaskApp/models/final_model4.pkl","rb",))



app = Flask(__name__)


@app.route('/', methods = ['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('main.html')

    if request.method == 'POST':
        #cat_cols
        region_of_moscow = request.form['region_of_moscow']
        #num_cols
        min_to_metro	= float(request.form['min_to_metro'])
        total_area	= float(request.form['total_area'])
        floor	= float(request.form['floor'])
        number_of_floors = float(request.form['number_of_floors'])
        construction_year = float(request.form['construction_year'])
        is_new = float(request.form['is_new'])
        is_apartments = float(request.form['is_apartments'])
        ceiling_height = float(request.form['ceiling_height'])
        number_of_rooms = float(request.form['number_of_rooms'])

        X_cat_from_form = [region_of_moscow]

        le_list = [region_of_moscow_LE]

        X_le_list = [] #под закодированные признаки

        for i in range(len(X_cat_from_form)):
            x_cat = le_list[i].transform([X_cat_from_form[i]])[0]
            # print(x_cat)
            X_le_list.append(x_cat)
        
        X_nums_from_form =[min_to_metro, total_area, floor, 
        number_of_floors, construction_year, is_new, is_apartments,
        ceiling_height, number_of_rooms]

        X = []
        X.extend(X_le_list)
        X.extend(X_nums_from_form)
        
        X_scaled = num_scaler.transform([X])

        prediction = model.predict(X_scaled)
        print(prediction)


        return render_template('main.html',
                               result = prediction)


@app.route("/api/v1/get_message/", methods=["POST", "GET"])
def api_message():
    get_message_x = request.json

    X_scaled = num_scaler.transform(get_message_x['X_for_predict'])
    print("X_scaled:", X_scaled)
    
    prediction = model.predict(X_scaled)

    return jsonify(str(prediction))



if __name__ == "__main__":
    app.run(debug=True)






    # X_cat_from_keyboard = ["СЗАО"]
    # # print(X_cat_from_keyboard)
    # le_list = [region_of_moscow_LE]

    # X_le_list = []  # под закодированные признаки

    # for i in range(len(X_cat_from_keyboard)):
    #     x_cat = le_list[i].transform([X_cat_from_keyboard[i]])[0]
    #     # print(x_cat)
    #     X_le_list.append(x_cat)
    # # print("X_cat_le:", X_le_list)

    # ##num
    # X_nums_from_keyboard = [5, 40, 5, 20, 2020, 1, 1, 3, 3]
    # # print("X_nums", X_nums_from_keyboard)

    # ##объединить категориальные и числовые (в том же порядке, как и при обучении)
    # X = []
    # X.extend(X_le_list)
    # X.extend(X_nums_from_keyboard)
    # # print("X:", X)

    # # scaler
    # X_scaled = num_scaler.transform([X])
    # # print("X_scaled:", X_scaled)

    # # predict
    # prediction = model.predict(X_scaled)
    # print(prediction)

    # result = prediction
    # print("Стоимость квартиры: ", result)

    # return f"Стоимость квартиры: {result}"



