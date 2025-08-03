# Automated identification of Smooth newt individuals based on spot patterns

This project aims to match photos of the same salamander at different points in time to eachother. We provide a RESTful API exposing this functionality, storing past images in a PostgreSQL database.

It was developed during the course of the Interdisciplinary Project of the Honours Programme at the University of Antwerp's Science faculty, by
- [Jesse Daems](https://github.com/SpeedyCodes) (Computer Science)
- [Rune De Coninck](https://github.com/RuneDC333) (Mathematics)

under the supervision of Em. prof. dr. Nick Schryvers.

For a full report including technical details and results, see [report_eng.pdf](report_eng.pdf).

See also the [companion app](https://github.com/SpeedyCodes/salamander-tracking-frontend).

## Setup
- Clone the repo 
- (optional) Clone the app repo, build it with `flutter build web` and copy `index.html` to `server/templates` and all other built files to `server/static`.
- Spin up a PostgreSQL DB called `salamanders`
- Fill out config.py
- Build the docker container
- Run the docker container
- Access the website/API at http://localhost:5000
