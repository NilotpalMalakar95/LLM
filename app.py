from flask import Flask, render_template, request, jsonify
from ice_breaker import get_wikipedia_data

app = Flask(__name__)


@app.route("/")
def index():
    print("Welcome to my WIki Data Gatherer...")


@app.route("/process", methods=["POST"])
def process():
    keyword = request.form["keyword"]
    information = get_wikipedia_data(keyword=keyword)

    return jsonify(information)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
