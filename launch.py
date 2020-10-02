from flask import Flask, render_template, request
from functions.NLPFunction import NLPPredict, NLPTrain
import json


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'

@app.route("/")
def home():
    return render_template("formulaire_upload.html", title='Home')


@app.route("/prediction", methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        user_text = request.form.get('user_text')
        if(user_text is None):
            prediction = ''
        else:
            prediction = str(NLPPredict(user_text)[0])
        return render_template("prediction_form.html", title='Prédiction', prediction=prediction)
    else:
        return render_template("prediction_form.html", title='Prédiction', prediction='')


@app.route("/entrainement", methods=['POST', 'GET'])
def entrainement():
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('pb1')
            score = str(NLPTrain())
            return render_template("entrainement_form.html", title='Entraînement', score=score)
        else:
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                print('pb2')
                score = str(NLPTrain())
                return render_template("entrainement_form.html", title='Entraînement', score=score)
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            score = str(NLPTrain(filename))

        return render_template("entrainement_form.html", title='Entraînement', score = score)
    else:
        return render_template("entrainement_form.html", title='Entraînement', score = False)

if __name__ == "__main__":
    app.run()

