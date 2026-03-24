from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    followers = int(request.form["followers"])
    following = int(request.form["following"])
    posts = int(request.form["posts"])
    username_length = int(request.form["username_length"])
    bio_length = int(request.form["bio_length"])
    profile_pic = int(request.form["profile_pic"])
    verified = int(request.form["verified"])

    data = np.array([[followers,following,posts,username_length,bio_length,profile_pic,verified]])

    prediction = model.predict(data)

    if prediction[0] == 1:
     result = "<span style='color:red;'>🚫 Fake Profile</span>"
    else:
      result = "<span style='color:green;'>✅ Real Profile</span>"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)