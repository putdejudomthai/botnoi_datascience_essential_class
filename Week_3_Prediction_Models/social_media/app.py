from flask import Flask, request, jsonify
import pandas as pd
import predict_social as social

app = Flask(__name__)

@app.route('/')
def index():
  return "Hello BOTNOI I'm Ro!"

@app.route('/socialtime')
def socialMedia ():
  gender = request.args.get('gender')
  age = request.args.get('age')
  mobile = request.args.get('mobile')
  relationship = request.args.get('relationship')

  # gender = 'ชาย'
  # age = '21'
  # mobile = 'Android'
  # relationship = 'มี'

  answer_list = [gender, age, mobile, relationship]
  pred = social.pred_pipeline(answer_list)
  pred = round(pred[0], 2)

  result = {'res': pred}

  return jsonify(result)

if __name__ == '__main__':
  app.run(debug=True)