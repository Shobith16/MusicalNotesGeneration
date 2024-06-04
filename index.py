from flask import Flask, render_template, jsonify, request ,Response
from werkzeug.utils import secure_filename
from predict import predict_instrument
from musictomid import pred_midifile
from mditonotes import pred_notes
from preprocess import mixed_sep
from Findemptyfiles import findempty
import os
import csv
from instrumentdetection import detect_vaildfile
from playmidi import play

app = Flask(__name__)
app.static_folder = 'static'
app.static_url_path = '/static'
# upload folder path
UPLOAD_FOLDER = 'D:/Final/Musical_Notes/separatedfiles/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
notes=[]
speratedfiles=[]

@app.route('/')
def index():
    return render_template('index.html')


    
@app.route('/upload', methods=['POST'])
def upload():
    # Check if "midiFile" exists in the uploaded files
    if "mixedFile" not in request.files:
        return jsonify({'Error': 'No file found'})
     
    # Get the uploaded file
    mixed_audio_file = request.files["mixedFile"]
    print(mixed_audio_file)
    filename = secure_filename(mixed_audio_file.filename)
    print(filename)
    # fname=filename
    

    if not (filename.endswith(".wav") or filename.endswith(".mp3")):
        return jsonify({'Error': 'Input file must be either a MP3 or WAV file'})

    

    # check=request.files["check"]
    check_string = request.form.get("check")

    # Convert string representation to boolean
    check_boolean = check_string.lower() == "true"
    print(check_boolean)
    valid_i=[]
    inst=[]

    if mixed_audio_file:
        
        mixed_audio_file.save(filename)
        
        if (check_boolean):
            speratedfiles=mixed_sep(filename)
            # valid_i=speratedfiles
            # check the file is empty or not
            valid_i=findempty(speratedfiles)
            print("after filtering empty audio files:",valid_i)
            
          
        else:
            # speratedfiles=filename
            # valid=detect_vaildfile(filename)
            valid="musical_instrument"
            if valid == "musical_instrument":
                predicted_instrument = predict_instrument(filename)

                valid_instruments = ['Sound_Piano','Sound_Guitar', 'Sound_Drum', 'Sound_Violin']

                # Check if the predicted instrument is valid
                if predicted_instrument in valid_instruments:
                    print("predicted instrument:",predict_instrument)
                    inst.append(predicted_instrument)
                    valid_i.append(filename)
            else:
                return jsonify({'Error': valid})        
        
        
        if len(valid_i)>0:  
            print("valid_sounds:",valid_i)      
            return jsonify({'sperated_f':valid_i,'check':check_boolean ,'inst':inst })
        else :
            print(len(valid_i))
            return jsonify({'Error':'Note Cannot be Generated for this audio!'})
           
    else:
        return jsonify({'Error':'no files'})
    

@app.route('/separatedfiles/<filename>/<check>')
def generate_notes(filename,check):
    # Log the request data for debugging
    # app.logger.info("Request Data: %s", request.form)

    # # Check if 'audioFile' exists in the uploaded files
    # if 'audioFile' not in request.files:
    #     app.logger.error("No 'audioFile' found in request")
    #     return jsonify({'error': 'No file found'})

    # audio_file = request.files['audioFile']
    audio_file =filename
    check_boolean=check.lower() =="true"
    print(audio_file,check_boolean)
    
    if audio_file == '':
        return jsonify({'Error': 'No selected file'})
    
    ins = []  # Initialize an empty list to store predicted instruments
    if audio_file:
        if check_boolean:
            # Predict the instrument only if check_boolean is true
            audio_file = os.path.join(app.config['UPLOAD_FOLDER'], audio_file)
            valid = detect_vaildfile(audio_file)
            if valid == "musical_instrument":
                predicted_instrument = predict_instrument(audio_file)
                valid_instruments = ['Sound_Piano', 'Sound_Guitar', 'Sound_Drum', 'Sound_Violin']
                if predicted_instrument in valid_instruments:
                    ins.append(predicted_instrument)  # Append the predicted instrument name to the list

        mid_file = pred_midifile(audio_file)
        notes =''
        print("previous :",notes)
        notes = pred_notes(mid_file)
        notes_string = ' '.join(notes)

        # Return the predicted notes and predicted instruments to the client
        response = jsonify({'Notes': notes_string, 'midi': mid_file, 'inst': ins})


        # Delete all files in the folder after sending the response
        # folder_path = app.config['UPLOAD_FOLDER']
        # for file_name in os.listdir(folder_path):
        #     file_path = os.path.join(folder_path, file_name)
        #     if os.path.isfile(file_path):
        #         os.remove(file_path)
        return response
        
    else:
        return jsonify({'Error': 'Invalid file format or file is empty'})



@app.route('/downloads', methods=['POST'])
def download_notes():
    data = request.get_json()
    notes_data = data.get('notes')
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400

    if not isinstance(notes_data, list):
        return jsonify({'error': 'Notes data must be a list'}), 400

    try:
        # Define column names
        column_names = ['Note']

        # Prepare CSV data with column names
        csv_data = ','.join(column_names) + '\n'
        csv_data += '\n'.join(notes_data)

        # Set content type and headers for file download
        headers = {
            'Content-Type': 'text/csv',
            'Content-Disposition': f'attachment; filename={filename}.csv'
        }

        # Create a Flask Response object with CSV data
        response = Response(csv_data, headers=headers)

        # Return the response for file download
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/play', methods=['POST'])
def play_Notes():
    midi_path = request.form.get('midiPath')
    print("Recived..",midi_path)
   
    # Assuming the play function accepts MIDI file content and returns a boolean indicating success
    success = play(midi_path)
    return jsonify({'success': success})





if __name__ == '__main__':
    app.run(debug=True)
