import pandas as pd
import pickle
from flask import Flask, render_template, request, send_from_directory, redirect, flash, url_for, jsonify, make_response, abort
from annotation_generator import AnnotationGenerator

app = Flask(__name__)

anno_generator_256ld = AnnotationGenerator(trained_model='OG_adam_100ep_256ld',
                                            latent_dim=256)
anno_generator_512ld = AnnotationGenerator(trained_model='rms_80ep_512ld',
                                            latent_dim=512)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/explanation', methods=['POST'])
def respond_to_explanation():
    if request.method == 'POST':
        print(request.method)
        print(request.form)

        lyrics = request.form['lyrics'] if len(request.form['lyrics']) > 0 else request.form['sample_lyrics']
        model = request.form['model']
        try:
            sentence = int(request.form['sentence']) / 50
            if model == 'Seq2Seq LSTM, 256 latent dimensions (better grammar, worse intuition)':
                explanation = anno_generator_256ld.reply(lyrics, diversity=True, temp=sentence)
            else:
                explanation = anno_generator_512ld.reply(lyrics, diversity=True, temp=sentence)
        except:
            if model == 'Seq2Seq LSTM, 256 latent dimensions (better grammar, worse intuition)':
                explanation = anno_generator_256ld.reply(lyrics, diversity=True)
            else:
                explanation = anno_generator_512ld.reply(lyrics, diversity=True)
        return render_template('explanation.html', lyrics=lyrics, explanation=explanation, sentence=sentence, model=model) #temperature=temperature, model=model)

@app.route('/contact', methods=['POST'])
def thank_contact():
    name = request.form['name'] if len(request.form['name']) > 0 else "blank_name"
    email = request.form['email'] if len(request.form['email']) > 0 else "blank_email"
    msg = request.form['comments'] if len(request.form['comments']) > 0 else "blank_comments"
    f = open("contact_info.txt", "a")
    f.write(name + '\n' + email + '\n' + msg + '\n\n')
    f.close()

    return render_template('contact.html', name=name)


if __name__ == '__main__':
    anno_generator_256ld.test_run()
    anno_generator_512ld.test_run()
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
