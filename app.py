from flask import Flask,render_template,url_for,request,redirect,send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from io import BytesIO
import os
import matplotlib.pyplot as plt
import pandas as pd
from my import get_shape,show_accuracy,set_get_X,set_get_Y,set_dataframe,remove_Col,classify_columns,split_train_test_scale
from my import model_LogisticRegression,model_KNeigh,model_SVM,model_kernelSVM,model_NaiveBayes,model_DecisionTree,model_RandomForest
#Defining App
app = Flask(__name__,static_url_path='/static')
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
#Creating Databases
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.String(120), unique=True, nullable=False)
    udata = db.relationship('Dataset',backref='author',lazy=True)
    def __repr__(self):
        return f"User('{self.username}', '{self.description}')"
class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ds_name = db.Column(db.String(40),unique=True,nullable=False)
    npc = db.Column(db.String(15),nullable=False)
    description = db.Column(db.String(100),nullable=False)
    date_uploaded = db.Column(db.DateTime,nullable=False,default=datetime.now())
    data = db.Column(db.LargeBinary,nullable=False)
    predict_data = db.Column(db.LargeBinary)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'),nullable=False)
    def __repr__(self):
        return f"User('{self.ds_name}','{self.npc}','{self.description}', '{self.date_uploaded}','{self.data}','{self.user_id})"


                                       #Handling Routes
#Home Page
@app.route('/', methods=["GET"])
def About():
    return render_template("cover.html")
#Getting-StartedPage
@app.route('/getting-started',methods=["GET","POST"])
def getting_started():
    theList =[]
    for user in User.query.all():
        if user not in theList:
            theList.append([user.username,user.description])
    return render_template("getting-started.html",userlist=theList)
#Adding New-User
@app.route('/add-new-user',methods=["GET","POST"])
def add_new_user():
    if request.method == "GET":
        return render_template("new-user.html")
    if request.method == "POST":
        firstname = request.form['firstname']
        lastname =request.form['lastname']
        description = request.form['description']
        user = User(username=firstname.capitalize()+" "+lastname.capitalize(),description=description)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("getting_started"))
#Existing User
@app.route('/existing-user/<exuser>',methods=['GET'])
def existing_user(exuser):
    thedlist=[]
    user = User.query.filter_by(username=exuser).first()
    for d in user.udata:
        if d not in thedlist:
            thedlist.append([d.ds_name])
    return render_template("exusds.html",user=exuser,dataname=thedlist)
# Adding Dataset
@app.route('/existing-user/<exuser>/add-dataset',methods=["GET","POST"])
def add_dataset(exuser):
    if request.method == "GET":
        return render_template("addDataset.html")
    if request.method == "POST":
        datasetname = request.form['nameofd']
        npc = request.form['npc']
        desciption = request.form['description']
        filedata = request.files['file']
        user_ref = User.query.filter_by(username=exuser).first()
        newFile = Dataset(ds_name=datasetname,npc=npc,description=desciption,data=filedata.read(),user_id=user_ref.id)
        db.session.add(newFile)
        db.session.commit()
        return redirect(url_for(".existing_user",exuser=exuser))
#Removing Dataset
@app.route('/existing-user/<user>/<dataname>/remove',methods=["GET"])
def remove_dataset(user,dataname):
    removing_set = Dataset.query.filter_by(ds_name=dataname).first()
    db.session.delete(removing_set)
    db.session.commit()
    return redirect(url_for(".existing_user",exuser=user))
#Visualizing the Data
@app.route('/existing-user/<user>/<dataname>/visualize',methods=["GET","POST"])
def visualize_data(user,dataname):
    filedata = BytesIO(Dataset.query.filter_by(ds_name=dataname).first().data)
    des = Dataset.query.filter_by(ds_name=dataname).first().description
    timestamp = Dataset.query.filter_by(ds_name=dataname).first().date_uploaded
    set_dataframe(filedata)
    remove_Col("id","Unnamed: 32")
    classify_columns()
    list1 = split_train_test_scale(set_get_X(),set_get_Y())
    Lg = show_accuracy(list1[3],model_LogisticRegression(list1[0],list1[2],list1[1]))
    KN = show_accuracy(list1[3],model_KNeigh(list1[0],list1[2],list1[1]))
    Svm = show_accuracy(list1[3],model_SVM(list1[0],list1[2],list1[1]))
    Ksvm = show_accuracy(list1[3],model_kernelSVM(list1[0],list1[2],list1[1]))
    NB = show_accuracy(list1[3],model_NaiveBayes(list1[0],list1[2],list1[1]))
    DT = show_accuracy(list1[3],model_DecisionTree(list1[0],list1[2],list1[1]))
    RF = show_accuracy(list1[3],model_RandomForest(list1[0],list1[2],list1[1]))
    shape = get_shape()
    loc = 'static/images/'
    # save_histplots(loc)
    # hists = os.listdir('./static/images/')
    return render_template("visualize.html",dataset=dataname.capitalize(),rows=shape[0],col=shape[1],description=des,dt=timestamp,LGA=Lg,NNA=KN,SVM=Svm,KSVM=Ksvm,NBA=NB,DTA=DT,RFA=RF)
#download-Dataset
@app.route('/existing-user/<user>/<dataname>/visualize/download-dataset',methods=["GET"])
def download_dataset(user,dataname):
    filedata = BytesIO(Dataset.query.filter_by(ds_name=dataname).first().data)
    ds = Dataset.query.filter_by(ds_name=dataname).first().ds_name
    return send_file(filedata,attachment_filename=ds+".csv",as_attachment=True)
#Predicting Data
@app.route('/existing-user/<user>/<dataname>/visualize/<algo>',methods=["GET","POST"])
def predict_LR(user,dataname,algo):
    if request.method == "POST":
        filedata = BytesIO(Dataset.query.filter_by(ds_name=dataname).first().data)
        set_dataframe(filedata)
        remove_Col("id","Unnamed: 32")
        classify_columns()
        list1 = split_train_test_scale(set_get_X(),set_get_Y())
        filedata1 = BytesIO(request.files['file'].read())
        datafr = pd.read_csv(filedata1)
        try:
            del datafr["Unnamed: 32"]
        except KeyError:
            print("Handled")
        Xy = datafr.iloc[:,:]
        if algo == "LR":
            Y_res = model_LogisticRegression(list1[0],list1[2],Xy)
            datafr["results"]=Y_res
            datafr.to_csv("final.csv")
            return send_file("final.csv",attachment_filename="pred_results.csv",as_attachment=True)
        elif algo == "KNC":
            Y_res = model_KNeigh(list1[0],list1[2],Xy)
            datafr["results"]=Y_res
            datafr.to_csv("final.csv")
            return send_file("final.csv",attachment_filename="pred_results.csv",as_attachment=True)
        elif algo == "SVM":
            Y_res = model_SVM(list1[0],list1[2],Xy)
            datafr["results"]=Y_res
            datafr.to_csv("final.csv")
            return send_file("final.csv",attachment_filename="pred_results.csv",as_attachment=True)
        elif algo == "KSVM":
            Y_res = model_kernelSVM(list1[0],list1[2],Xy)
            datafr["results"]=Y_res
            datafr.to_csv("final.csv")
            return send_file("final.csv",attachment_filename="pred_results.csv",as_attachment=True)
        elif algo == "NB":
            Y_res = model_NaiveBayes(list1[0],list1[2],Xy)
            datafr["results"]=Y_res
            datafr.to_csv("final.csv")
            return send_file("final.csv",attachment_filename="pred_results.csv",as_attachment=True)
        elif algo == "DT":
            Y_res = model_DecisionTree(list1[0],list1[2],Xy)
            datafr["results"]=Y_res
            datafr.to_csv("final.csv")
            return send_file("final.csv",attachment_filename="pred_results.csv",as_attachment=True)
        elif algo == "RFC":
            Y_res = model_RandomForest(list1[0],list1[2],Xy)
            datafr["results"]=Y_res
            datafr.to_csv("final.csv")
            return send_file("final.csv",attachment_filename="pred_results.csv",as_attachment=True)
    if request.method == "GET":
        return render_template("predict.html")
#Scatter Plot Graph
@app.route('/existing-user/<user>/<dataname>/visualize/graphs',methods=["GET","POST"])
def graph_plot(user,dataname):
    filedata = BytesIO(Dataset.query.filter_by(ds_name=dataname).first().data)
    x = set_dataframe(filedata)
    if request.method == "GET":
        return render_template("getCol.html")
    if request.method == "POST":
        col1 = request.form["Col1"]
        col2 = request.form["Col2"]
        plt.scatter(x=x.loc[:,col1].values,y=x.loc[:,col2].values,c='DarkBlue')
        plt.title(col1+" vs "+col2)
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.savefig("static/images/"+col1+"_vs_"+col2+".jpg")
        return render_template("iplot.html",s=col1+"_vs_"+col2+".jpg",col1=col1,col2=col2)
if __name__ == '__main__':
    db.create_all()
    app.debug = True
    app.run(host="0.0.0.0",port=5609)
