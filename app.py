from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
from src.pipeline.training_pipeline import train_model

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('form.html')
    else:
        data=CustomData(
            LIMIT_BAL=float(request.form['limit_bal']),
            SEX=int(request.form['sex']),
            EDUCATION=int(request.form['education']),
            MARRIAGE=int(request.form['marriage']),
            AGE=int(request.form['age']),
            PAY_1=int(request.form['pay_1']),
            PAY_2=int(request.form['pay_2']),
            PAY_3=int(request.form['pay_3']),
            PAY_4=int(request.form['pay_4']),
            PAY_5=int(request.form['pay_5']),
            PAY_6=int(request.form['pay_6']),
            BILL_AMT1=float(request.form['bill_amt1']),
            BILL_AMT2=float(request.form['bill_amt2']),
            BILL_AMT3=float(request.form['bill_amt3']),
            BILL_AMT4=float(request.form['bill_amt4']),
            BILL_AMT5=float(request.form['bill_amt5']),
            BILL_AMT6=float(request.form['bill_amt6']),
            PAY_AMT1=float(request.form['pay_amt1']),
            PAY_AMT2=float(request.form['pay_amt2']),
            PAY_AMT3=float(request.form['pay_amt3']),
            PAY_AMT4=float(request.form['pay_amt4']),
            PAY_AMT5=float(request.form['pay_amt5']),
            PAY_AMT6=float(request.form['pay_amt6'])
            

)
        final_new_data=data.get_data_as_dataframe()
        if None in final_new_data.values:
            print('None values found in data')
            print(final_new_data.values)
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)

@app.route('/train',methods=['GET','POST'])
def train():
    report=train_model()
    model_name='LogisticRegression'
    final={'report':report,'model_name':model_name}
    return render_template('train_model.html',final=final)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

