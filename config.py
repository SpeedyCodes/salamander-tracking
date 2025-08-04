PG_CONNECTION_STRING = "postgresql://postgres:postgres@salamanders_db:5432/salamanders"
huey_immediate = True
pose_estimation_timeout = 10
jwt_secret = "quite-secure" # can be any string, but the more random the better
minimum_required_score = 0.4
pose_estimation_confidence = 0.6
# this password will be set if no other password exists in the database
default_password = "very-secure" # you can make this empty again after first startup