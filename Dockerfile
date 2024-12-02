FROM python:3.11
LABEL authors="Jesse Daems & Rune De Coninck"

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn
RUN apt update
RUN apt install libopencv-dev -y
RUN python -c "import largestinteriorrectangle"

COPY server/ server/
COPY src/ src/
COPY training/haar_cascade training/haar_cascade
COPY training/dlc/salamander-jesse-2024-09-15 training/dlc/salamander-jesse-2024-09-15
COPY config.py config.py
COPY migrations/ migrations/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]

CMD ["gunicorn", "server.wsgi:app", "-b", "0.0.0.0:5000"]
