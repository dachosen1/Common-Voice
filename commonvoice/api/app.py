from flask import Flask, render_template
import markdown
import os
from commonvoice.api.config import PACKAGE_ROOT

app = Flask(
    __name__,
    static_folder="css",
    static_url_path="/css",
    template_folder="templates",
)


@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    with open(os.path.join(PACKAGE_ROOT.parent, 'README.md'), "r") as markdown_file:
        content = markdown_file.read()
        return markdown.markdown(content)
