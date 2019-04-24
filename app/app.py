'''
Currently, this code is an example from the Fraud Detection Case Study...
'''
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



#
#     return render_template('home.html')
#
# @app.route('/about')
# def about():
#     return 'About Us'
#
# @app.route('/explanation', methods=['POST', 'GET'])
# def generate_annotation():
#     if request.method == 'POST':
#         if 'sentence' not in request.form:
#             flash('No sentence post')
#             redirect(request.url)
#         elif request.form['sentence'] == '':
#             flash('No sentence')
#             redirect(request.url)
#         else:
#             sent = request.form['sentence']
#             annotations_generated.append('LYRIC: ' + sent)
#             reply = annotation_generator.reply(sent)
#             annotations_generated.append('ANNOTATION: ' + reply)
#     return render_template('explain_lyrics.html', conversations=annotations_generated)





    # TRAINED_MODEL_PATH = 'models/final_model.h5'
    # this is where i should load in the different dictionaries, etc., necessary for pred
    # with open ('models/class_names.pkl', 'rb') as f:
    #     class_names = np.array(pickle.load(f))

    # Load trained model
    # Since I used a custom optimizer, I have to define and load it here
    # trained_model = load_model(TRAINED_MODEL_PATH, custom_objects={'trained_model': trained_model})
    #
    # generator = AnnotationGenerator(use_weights=True, trained_model='trained_model', data_name='baseline_data', latent_dim=512)
    # generator._chat_over_command_line()

    # print('Model loaded. Start serving...')


    # annotation_generator.test_run()
    # app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)



# def recent_fraud():
#     # mongo client stuff
#     X_test = pd.read_csv('data/x_target.csv')
#     y_test = pd.read_csv('data/y_target.csv')
#     loaded_model = pickle.load(open('data/pkl_for_derek.pkl', 'rb'))
#     result = loaded_model.predict_proba(X_test.iloc[90:100])
#     actual = y_test.iloc[90:100]
#     predicted = [round(i[1],3) for i in result]
#     threshold = 0.06
#     prediction = []
#     results = []
#     for i in result:
#         if i[1] < threshold:
#             prediction.append(['Not Fraud'])
#         else:
#             prediction.append(['Fraud'])
#     return render_template('recent_fraud.html',
#         data = zip(predicted, prediction, actual.values[:,1]))





# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']
#
#
#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         # f.save(file_path)
#         rotate_save(f, file_path)
#
#         # Make prediction
#         preds = model_predict(file_path, model)
#
#         # Delete it so we don't clutter our server up
#         os.remove(file_path)
#
#         return preds
#     return None
