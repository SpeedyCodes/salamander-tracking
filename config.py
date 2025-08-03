PG_CONNECTION_STRING = "postgresql://[PG_USERNAME_HERE]:[PG_PASSWORD_HERE]@localhost:5432/salamanders"
huey_immediate = True
pose_estimation_timeout = 10
jwt_secret = "" # can be any string, but the more random the better
minimum_required_score = 0.4
pose_estimation_confidence = 0.6

assert jwt_secret != "" and "HERE" not in PG_CONNECTION_STRING, "Please fill out config.py before continuing"