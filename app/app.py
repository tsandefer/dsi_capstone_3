import pandas as pd
import pickle
from flask import Flask, render_template, request, send_from_directory, redirect, flash, url_for, jsonify, make_response, abort
from annotation_generator import AnnotationGenerator

app = Flask(__name__)

filepath = '../../cap3_models/models/'
name = 'baseline_rms_800ep_512ld'
weights = '-weights_final'
annotation_generator = AnnotationGenerator(use_weights=True, models_filepath=filepath, data_filepath=filepath,
                            trained_model=name, data_name=name, final_weights_fp=weights,
                            latent_dim=512)

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        print(request.method)
        print(request.form)

        if 'lyrics' in request.form:
            lyrics = request.form['lyrics']
            model = request.form['model']
            try:
                sentence = request.form['sentence']
                explanation = annotation_generator.reply(lyrics, diversity=True, temp=sentence)
            except:
                explanation = annotation_generator.reply(lyrics, diversity=True)
            return render_template('explanation.html', lyrics=lyrics, explanation=explanation, model=model, sentence=sentence) #temperature=temperature, model=model)


    elif request.method == 'GET':
        return render_template('home.html')

@app.route('/explanation', methods=['POST', 'GET'])
def respond_to_explanation():
    if request.method == 'POST':
        print(request.method)
        print(request.form)

        # if 'sentence' in request.form:
        #     lyrics = request.form['sentence']
        #     explanation = annotation_generator.reply(lyrics, diversity=True, temp=0.65)
        return render_template('home.html')

    # elif request.method == 'GET':
        # return redirect(url_for('try_again')
        # return render_template('explanation.html', lyrics=lyrics, explanation=explanation)


def main():
    annotation_generator.test_run()
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
