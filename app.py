from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('ipl_score_predictor.lb')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contacts')
def contacts():
    return render_template('contacts.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data and set placeholder values for missing columns
    data = {
        'venue': [request.form['venue']],
        'bat_team': [request.form['bat_team']],
        'bowl_team': [request.form['bowl_team']],
        'batsman': [request.form['batsman']],
        'bowler': [request.form['bowler']],
        'runs': [int(request.form['runs'])],
        'wickets': [int(request.form['wickets'])],
        'overs': [int(request.form['overs'])],
        'runs_last_5': [int(request.form['runs_last_5'])],
        'wickets_last_5': [int(request.form['wickets_last_5'])],
        'striker': [int(request.form['striker'])],
        'non-striker': [int(request.form['non_striker'])],
        'day': [int(request.form['day'])],  
        'year': [int(request.form['year'])],  
        'month': [int(request.form['month'])], 
        'mid': [0]  
    }
    
    input_data = pd.DataFrame(data)
    
    print(input_data.columns)
    
    try:
        prediction = model.predict(input_data)[0]
    except ValueError as e:
        print(f"Error: {e}")
        prediction = "Error: Unable to make prediction. Check input data."
    
    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
