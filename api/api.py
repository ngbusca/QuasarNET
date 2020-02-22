import flask
from flask import request, render_template
import sys
import numpy as np

from quasarnet import io
from quasarnet.utils import process_preds
from tensorflow.keras.models import load_model
from quasarnet.models import custom_loss

app = flask.Flask(__name__)
app.config["DEBUG"] = True
model = load_model(sys.argv[1], custom_objects = {'custom_loss':custom_loss})
lines = ['LYA','CIV(1548)','CIII(1909)','MgII(2796)','Hbeta','Halpha']
lines_bal = ['CIV(1548)']

@app.route('/predict', methods=['POST'])
def predict():
    cl = request.content_length
    if cl is not None and cl > 20 * 1024 * 1024:
        return "input file too big (limit 20MB)"

    req_data = request.get_json()
    fl = np.array(req_data['flux']).astype(float)
    we = np.array(req_data['ivar']).astype(float)

    if 'wave' in req_data:
        ll = np.log10(np.array(req_data['wave']).astype(float))
        ## binning matrix
        B = (np.log10(io.wave[:,None])-ll[None,:])/io.dll
        w = (B>0) & (B<1)
        B[w] = 1.
        B[~w] = 0.

        fl_qn = []
        we_qn = []

        for i in range(len(fl)):
            fl_aux = B.dot(fl[i]*we[i])
            we_aux = B.dot(we[i])
            w = we_aux > 0
            fl_aux[w]/=we_aux[w]
            fl_qn.append(fl_aux)
            we_qn.append(we_aux)

        fl = np.array(fl_qn)
        we = np.array(we_qn)

    if we is None:
        we = fl*0 + 1.
    X = fl-np.average(fl, weights=we, axis=1)[:,None]

    norm = X.std(axis=1)
    w = norm>0
    X[w]/= norm[w,None]
    
    p = model.predict(X[:,:,None])
    p_line,z_line,zbest,p_bal,z_bal = process_preds(p, lines, lines_bal)
    ret = {"zbest": [z for z in zbest], "lines":lines, "lines_BAL":lines_bal}
    print(len(p_line))
    for i,l in enumerate(lines):
        ret["p_{}".format(l)] = [str(p) for p in p_line[i]]
        ret["z_{}".format(l)] = [str(z) for z in z_line[i]]
    for i, l in enumerate(lines_bal):
        ret["pBAL_{}".format(l)] = [str(p) for p in p_bal[i]]
        ret["zBAL_{}".format(l)] = [str(z) for z in z_bal[i]]

    return ret

if __name__ == '__main__':
    app.run(host='0.0.0.0')
