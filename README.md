## _Everything Looks Like Chicken (and Waffles)_
### A Neural Image Captioning System

** UPDATE ** - The application demo is no longer live. See the instructions below on how to download it locally. You may also contact me, and I can quickly get the demo up and running again on a public server. 

---

This project is a submission to the <a href="https://www.yelp.com/dataset/challenge"> Yelp Dataset Challenge </a>. Using <a href="https://www.yelp.com/dataset/challenge"> Yelp's public dataset</a>,
I trained a neural network that is designed to comment on pictures of food, as humans do on Yelp's site. 

The system <strike>is</strike> **was** live <a href="http://ec2-54-158-215-211.compute-1.amazonaws.com:5000" target="_blank"> <b>here</b></a> - check it out! If you're interested in how it works, be sure to check out the <a href="http://ec2-54-158-215-211.compute-1.amazonaws.com:5000/static/main.pdf" target="_blank"> <b> technical report</b></a> I wrote to accompany this project.

The model is built in `keras`, and the application is served using the `flask` microframework. 

---

If you'd like to run a demo locally, clone the the `server` directory and enter that directory. You'll then need to install the application's software requirements.

- Open a `virtualenv` (optional)
```bash
# start an environment
virtualenv your-env
# activate it
source your-env/bin/activate
```

- Install the required packages (python3)

```
python3 -m pip install -r requirements.txt
```

- Then, run the application
```bash
python3 app.py
```

And you're done! go to your localhost, port `5000`, and you'll see the application is live. 
