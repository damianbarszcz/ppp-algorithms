from flask import Flask
from models.db import init_db
from routes import positioning, mentors

app = Flask(__name__)

# Inicjalizacja bazy danych
app.config.from_pyfile('config.py')
init_db(app)

# Rejestracja tras
app.register_blueprint(positioning.bp)
app.register_blueprint(mentors.bp)

if __name__ == '__main__':
    app.run(debug=True)