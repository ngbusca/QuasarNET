import flask
from flask import request, render_template
import numpy as np

from quasarnet import io
from quasarnet.utils import process_preds
from tensorflow.keras.models import load_model
from quasarnet.models import custom_loss

app = flask.Flask(__name__)
app.config["DEBUG"] = True
model = load_model('QuasarNET/weights/qn_train_0.h5', custom_objects = {'custom_loss':custom_loss})

@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json()
    fl = np.array(req_data['flux'])
    we = np.array(req_data['weights'])
    if we is None:
        we = fl*0 + 1.
    X = fl-np.average(fl, weights=we)
    X /= X.std()
    
    p = model.predict(X[None,:,None])
    _,_,zbest,_,_ = process_preds(p,['LYA','CIV(1548)','CIII(1909)',
        'MgII(2796)','Hbeta','Halpha'],['CIV(1548)'])
    return "{}".format(zbest[0])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
