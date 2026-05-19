from flask import Flask, render_template, request
import pickle
import numpy as np
import json

app = Flask(__name__)

with open('all_models.pkl', 'rb') as file:
    trained_models = pickle.load(file)

try:
    with open('item_id_map.json', 'r') as file:
        accurate_item_map = json.load(file)
except FileNotFoundError:
    accurate_item_map = {}

model_metrics = [
    {"name": "Gradient Boosting", "train_r2": 0.6408, "test_r2": 0.5990, "rmse": 1113.57, "best": True},
    {"name": "LightGBM", "train_r2": 0.7464, "test_r2": 0.5870, "rmse": 1130.06, "best": False},
    {"name": "CatBoost", "train_r2": 0.7834, "test_r2": 0.5828, "rmse": 1135.74, "best": False},
    {"name": "Random Forest", "train_r2": 0.9379, "test_r2": 0.5767, "rmse": 1144.12, "best": False},
    {"name": "XGBoost", "train_r2": 0.8868, "test_r2": 0.5241, "rmse": 1213.06, "best": False}
]

fat_content_map = {'Low Fat': 0, 'Regular': 1}

item_type_map = {
    'Baking Goods': 0, 'Breads': 1, 'Breakfast': 2, 'Canned': 3, 
    'Dairy': 4, 'Frozen Foods': 5, 'Fruits and Vegetables': 6, 'Hard Drinks': 7, 
    'Health and Hygiene': 8, 'Household': 9, 'Meat': 10, 'Others': 11, 
    'Seafood': 12, 'Snack Foods': 13, 'Soft Drinks': 14, 'Starchy Foods': 15
}

outlet_id_map = {
    'OUT010': 0, 'OUT013': 1, 'OUT017': 2, 'OUT018': 3, 'OUT019': 4, 
    'OUT027': 5, 'OUT035': 6, 'OUT045': 7, 'OUT046': 8, 'OUT049': 9
}

outlet_size_map = {'High': 0, 'Medium': 1, 'Small': 2}
location_type_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}

outlet_type_map = {
    'Grocery Store': 0, 
    'Supermarket Type1': 1, 
    'Supermarket Type2': 2, 
    'Supermarket Type3': 3
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = ""
    error_message = ""
    selected_algo = "Gradient Boosting"
    
    if request.method == 'POST':
        selected_algo = request.form['model_name']
        
        raw_item_id = request.form['item_identifier'].strip().upper() 
        
        if raw_item_id in accurate_item_map:
            item_identifier = float(accurate_item_map[raw_item_id])
            
            item_weight = float(request.form['item_weight'])
            item_fat_content = float(fat_content_map.get(request.form['item_fat_content'], 0))
            item_visibility = float(request.form['item_visibility'])
            item_type = float(item_type_map.get(request.form['item_type'], 0))
            item_mrp = float(request.form['item_mrp'])
            outlet_identifier = float(outlet_id_map.get(request.form['outlet_identifier'], 0))
            outlet_year = float(request.form['outlet_year'])
            outlet_size = float(outlet_size_map.get(request.form['outlet_size'], 1))
            outlet_location = float(location_type_map.get(request.form['outlet_location'], 0))
            outlet_type = float(outlet_type_map.get(request.form['outlet_type'], 1))
            
            input_data = np.array([[
                item_identifier, item_weight, item_fat_content, item_visibility, 
                item_type, item_mrp, outlet_identifier, outlet_year, 
                outlet_size, outlet_location, outlet_type
            ]]) 
            
            model = trained_models[selected_algo]
            prediction = model.predict(input_data)
            
            final_prediction = max(0.0, prediction[0])
            prediction_text = f"${final_prediction:,.2f}"
        else:
            error_message = f"❌ Invalid Item Identifier '{raw_item_id}'! Please enter a valid product code (e.g., FDA15, DRC01)."
        
    return render_template('predict.html', 
                           prediction_text=prediction_text, 
                           error_message=error_message, 
                           selected_algo=selected_algo, 
                           model_metrics=model_metrics)

if __name__ == "__main__":
    app.run(debug=True)