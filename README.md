# Automated identification of Smooth newt individuals based on spot patterns

This project aims to match photos of the same salamander at different points in time to eachother. We provide a RESTful API exposing this functionality, storing past images in a PostgreSQL database.

It was developed during the course of the Interdisciplinary Project of the Honours Programme at the University of Antwerp's Science faculty, by
- [Jesse Daems](https://github.com/SpeedyCodes) (Computer Science)
- [Rune De Coninck](https://github.com/RuneDC333) (Mathematics)

under the supervision of Em. prof. dr. Nick Schryvers.

For a full report including technical details and results, see [report_eng.pdf](report_eng.pdf) or [report_dutch.pdf](report_dutch.pdf).

See also the [companion app](https://github.com/SpeedyCodes/salamander-tracking-frontend).

## Setup
- Clone the repo 
- (optional) Clone the app repo, build it with `flutter build web` and copy `index.html` to `server/templates` and all other built files to `server/static`.
- (optional) Fill out config.py with custom settings

To run the software, there are three options. All three should eventually make the 
website/API accessible at http://localhost:5000 .
### Run with Docker Compose
This is the easiest option, if you have Docker Compose: `docker compose up` should do the trick.
### Run with just Docker and your own PostgreSQL database
Spin up your database named `salamanders`, and edit the connection string in `config.py` accordingly.
Next, just build the docker container and run it.
### Run with Python 3.11.0 and your own PostgreSQL database
Spin up your database named `salamanders`, and edit the connection string in `config.py` accordingly.
Next, run these commands:
- `pip install -r requirements.txt` to install dependencies
- `python -m server.app` to run the server

