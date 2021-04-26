import os
from flask import Flask,flash,redirect,render_template,url_for,request,jsonify,make_response,Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_login import LoginManager 
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required,  current_user
from werkzeug.utils import secure_filename
import pandas as pd
# import nummpy as np
import pickle
from flask_cors import CORS, cross_origin

import time

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

from sklearn.metrics import recall_score 




from flask_cors import CORS, cross_origin
# app = Flask(__name__)

 
app = Flask(__name__) 
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
	
UPLOAD_FOLDER='/Users/avanigoyal/Flask_tutorial/dish-tagger1/files' 
ALLOWED_EXTENSIONS = set(['xlsx', 'csv','xls'])
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:Avani@1998@localhost:3306/task1db'
app.config['SECRET_KEY']='secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TESTING'] = False
db = SQLAlchemy(app)

db.create_all()
from models import *

  
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

model_tf = pickle.load(open('TFIDF.pkl', 'rb'))
# print(model_tf.get_feature_names())
model_nb= pickle.load(open('NB.pkl', 'rb'))

model_tf_c = pickle.load(open('TFIDF_cat.pkl', 'rb'))
model_nb_c= pickle.load(open('SVM_cat.pkl', 'rb'))
print("*(************************************************************************************")
# print(current_user.email)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
    


@app.route('/')
def login():
    return render_template('base.html')

@app.route('/login', methods=['POST'])
def login_post():
    
    email = request.form.get('email')
    password = request.form.get('pswd')
   

    user = User.query.filter_by(email=email).first()
    print(user)
   
    if not user or not check_password_hash(user.password, password):
        res=make_response(jsonify({'error':'Check your email or password'}))
        return res
        # flash("Check your username or password")
        # return redirect(url_for('login'))

    # flash('You are logged in successfully')
    login_user(user)
    print(current_user.id)
    
    # return redirect(url_for('profile'))
    response = make_response(redirect(url_for('profile')))
    return response

@app.route('/signup')
def signup():
     return render_template('signup.html')



@app.route('/signup', methods=['POST'])
def signup_post():
    
    email = request.form.get('email')
    uname = request.form.get('uname')
    password = request.form.get('pswd')

    user = User.query.filter_by(email=email).first() 
    print(user)
    if user: 
        return jsonify({'error':'User already exist'})
        # flash("User already exist")
        # return redirect(url_for('login'))

    new_user = User(email=email, uname=uname, password=generate_password_hash(password, method='sha256'))
    
    db.session.add(new_user)
    db.session.commit()
    time.sleep(2)

   
    login_user(new_user)
    # return redirect(url_for('profile'))
    response = make_response(redirect(url_for('profile')))
    return response

@app.route('/logout')
@login_required
def logout():
    logout_user()
     
    response = make_response(redirect(url_for('login')))
    return response


@app.route('/profile', methods=['GET'])
@login_required
def profile():


    return render_template('profile.html')
    

@app.route('/jsonres')
@login_required
def jsonres():
    print("$$$$$$$")
    dish = DishData.query.filter_by(user_id=current_user.id).all()
    df = pd.DataFrame([(d.dname, d.price, d.cname,d.vnv) for d in dish], 
              columns=['dishname', 'price', 'category','vnv'])
    
    n = 10  #chunk row size
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]
    size=len(list_df)

    print(size)

    for i in list_df:
        print(len(i))

    if request.method=='POST':
        pno=request.get_json()
        p=pno['pgno']
        if(p>size-1):
            return jsonify({'error':'There is no more data to show'})
        # sno=str(pno.values())
        # num=int(sno)
        # print(num-1)
        opage=list_df[p]
        d=opage.to_dict(orient="records")
        j1=jsonify(data=d)
        j1.status_code = 200
        print(j1)
        res=make_response(j1)
        print("$$$$$$$$$$$$$$$$$")
        print(res)
        return res


    if size==0:
        return jsonify({'msg':'welcome to our website'})
    
    fpage=list_df[0]
    d=fpage.to_dict(orient="records")
    # print(d)
    j1=jsonify(data=d)
    j1.status_code = 200
    print(j1)
    res=make_response(j1)
    print("$$$$$$$$$$$$$$$$$")
    print(res)
    return res





@app.route('/upload')
@login_required
def upload():
    return render_template('data_entry.html')




def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/fileup', methods=['POST','GET'])
@login_required
def fileup():
    
    if request.method=='POST':
        file = request.files.get('file')
        print(file)
        if 'file' not in request.files:
            res=make_response(jsonify({'error':'select file to upload'}))
            return res
    
        if file.filename == '':
            res=make_response(jsonify({'error':'select file to upload'}))
            return res
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            print(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            

            

            # generate absolute file path
            file_path = os.path.join(UPLOAD_FOLDER, filename)

         

            #   add entry into database
            file_entry = FileData(file_name=filename,file_path=file_path,user_id=current_user.id)

            db.session.add(file_entry)
            db.session.commit()


            if filename.rsplit('.', 1)[1].lower()=='csv':
                try:
                    df=pd.read_csv(file_path,sep=',')
                    s=df.size
                    if s==0:
                        res=make_response(jsonify({'error':'your csv file is empty'}))
                        return res
                    print(s)
                    print(df)
                    for i,row in df.iterrows():
                        record=(list(row))
                        item,pr=record

                        x=[item]
    

                        pred_tf_c=model_tf_c.transform(x)
                        pred_nb_c=model_nb_c.predict(pred_tf_c)
                        cat=str(pred_nb_c[0])

                        pred_tf=model_tf.transform(x)
                        # print(pred_tf)
                        pred_nb=model_nb.predict(pred_tf)
                        v_nv=str(pred_nb[0])
                        
                        new_dish = DishData(dname=item, price=pr, cname=cat, vnv=v_nv, user_id=current_user.id)
                        db.session.add(new_dish)
                        db.session.commit()


                    # flash("Your data uploaded successfully")
                    res=make_response(redirect(url_for('upload')))
                    return res
                    # return redirect(url_for('upload'))

                except pd.errors.EmptyDataError:

                    # flash("Your csv file is empty.")
                    res=make_response(redirect(url_for('upload')))
                    return res


                    
                   
                


            if filename.rsplit('.', 1)[1].lower()=='xlsx' or filename.rsplit('.', 1)[1].lower()=='xls':
                df = pd.read_excel(file_path)
                if df.empty==True:

                    print(df)
                    res=make_response(jsonify({'error':'your excel file is empty'}))
                    return res

                for i,row in df.iterrows():
                    record=list(row)
                    item,pr=record
                    x=[item]
    

                    pred_tf_c=model_tf_c.transform(x)
                    pred_nb_c=model_nb_c.predict(pred_tf_c)
                    cat=str(pred_nb_c[0])

                    pred_tf=model_tf.transform(x)
                    # print(pred_tf)
                    pred_nb=model_nb.predict(pred_tf)
                    v_nv=str(pred_nb[0])


                    new_dish = DishData(dname=item, price=pr, cname=cat, vnv=v_nv, user_id=current_user.id)
                    db.session.add(new_dish)
                    db.session.commit()
                # flash("Your data uploaded successfully")
                # res=make_response(jsonify({'success':'your data uploaded'}))
                # return res  
                res=make_response(redirect(url_for('upload')))
                return res

                 
                    
    res=make_response(redirect(url_for('profile')))
    return res


@app.route('/add_dish', methods=['POST'])
@login_required
def add_dish():

    dname = request.form.get('dname')
    price = float(request.form.get('price'))
    
    x=[dname]
    

    pred_tf_c=model_tf_c.transform(x)
    pred_nb_c=model_nb_c.predict(pred_tf_c)
    cname=str(pred_nb_c[0])

    pred_tf=model_tf.transform(x)
    # print(pred_tf)
    pred_nb=model_nb.predict(pred_tf)
    vnv=str(pred_nb[0])



    # cname = request.form.get('cname')
    # vnv = request.form.get('vnv')
    #print(dname)
    new_dish = DishData(dname=dname, price=price, cname=cname, vnv=vnv, user_id=current_user.id)
    # cname = request.form.get('cname')
    # vnv = request.form.get('vnv')
    # # #print(dname)
    # new_dish = DishData(dname=dname, price=price, cname=cname, vnv=vnv, user_id=current_user.id)
    db.session.add(new_dish)
    db.session.commit()
    # flash("Your data added successfully")
    # return redirect(url_for('upload'))

    res=make_response(redirect(url_for('upload')))
    return res

@app.route("/hello")
# @cross_origin
@login_required
def hello():	
	print(current_user.id)
	return "hello world"

if __name__=='__main__':
    app.run(debug=True)    
        


              
                

                
                
                        