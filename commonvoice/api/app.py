from flask import Flask, render_template

app = Flask(__name__,
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


