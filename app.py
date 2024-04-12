from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)
table_html = None
@app.route('/')
def display_dataset():
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('data_set.csv')
    global table_html 
    # Convert the DataFrame to HTML table
    table_html = df.to_html(index=False)

    # Render the HTML template with the table data
    return render_template('index.html', table=table_html)



# Load the trained model
model = load('model.joblib')
df = pd.read_csv('preprocessed_data.csv')  # Load your dataset

# Define the list of features used for prediction
features = [col for col in df.columns if col not in ['Metric tons of CO2e per capita (2000)']]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        country_id = int(request.form['country'])  # Assuming country ID is provided as input
        year = int(request.form['year'])

        # Find the row corresponding to the input country ID
        filtered_df = df[df['Country_Encoded'] == country_id]
        
        if filtered_df.empty:
            return render_template('index.html', error_message="Country ID not found")
        
        country_encoded = filtered_df.index[0]  # Assuming there's exactly one match

        # Fetch the relevant features for the country
        country_data = df.iloc[country_encoded][features].values.reshape(1, -1)

        # Make prediction for the specific country and year
        predicted_co2 = model.predict(country_data)[0]

        return render_template('index.html', country=country_id, year=year, predicted_co2=predicted_co2, table=table_html)

if __name__ == '__main__':
    app.run(debug=True)
