import base64
from io import BytesIO
from PIL import Image
# import your_ml_model  # Import your ML model here
from equation_solver.pipeline.prediction import *
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
# app = Flask(__name__, static_url_path='', static_folder='static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/solve-equation', methods=['POST'])
def solve_equation():
    data = request.json
    image_data = data['image'].split(',')[1]  # Remove the "data:image/png;base64," part
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    
    # Save the image temporarily if needed
    # image.save('temp_equation.png')
    img_arr = np.array(image.convert('RGB'))
    
    # Use ML model to solve the equation
    # solution = your_ml_model.solve(image)
    # equation ,  final_equation , final_ans = process('temp_equation.png')
    equation ,  final_equation , final_ans = process(img_arr)
    solution = final_equation +" =  " +str(round(final_ans,2))
    
    return jsonify({'solution': solution})

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    # os.system("python main.py")
    return "Please train the model offline"
    # os.system("dvc repro")
    return "Training done successfully!"

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8080) #for AWS