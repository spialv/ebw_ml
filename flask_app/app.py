import flask
from flask import render_template
import pickle
import numpy as np
import scipy
import gunicorn

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['POST', 'GET'])

@app.route('/index', methods=['POST', 'GET'])
def main():
    with open('../models/ebw_gb_model.pkl', 'rb') as f:
        model = pickle.load(f)

    if flask.request.method == 'GET':
        return render_template('index.html', iw=47, ifoc=139, vf=4.5, fp=80, depth=1.60, width=2.54)

    if flask.request.method == 'POST':
        # запросить форму
        try:
            iw = int(flask.request.form['IW'])
            ifoc = int(flask.request.form['IF'])
            vf = float(flask.request.form['VF'])
            fp = int(flask.request.form['FP'])
        except ValueError:
            return render_template('index.html', depth="Неверный формат входных данных", width=":(")
        # сделать прогноз
        x = np.asarray([iw, ifoc, vf, fp]).reshape(1, -1)
        predicts = model.predict(x)
        # вернуть предсказание
        return render_template('index.html', depth=predicts[0][0].round(2), width=predicts[0][1].round(2),
                               iw=iw, ifoc=ifoc, vf=vf, fp=fp)


if __name__ == '__main__':
    app.run()
