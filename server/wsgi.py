from server.app import app, set_default_password
with app.app_context():
    set_default_password()
if __name__ == "__main__":
    app.run()