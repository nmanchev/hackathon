import ray
import os
import smtplib
import sys

from xgboost_ray import RayDMatrix, RayParams, train
from sklearn.datasets import load_breast_cancer
from domino import Domino
from email.mime.text import MIMEText



# Get command line arguments
if len(sys.argv)<2:
    print("Command line argument not provided. Please provide a float value to be used as learning rate.")
    print("Example: python train.py 0.1")
    sys.exit()

lr = float(sys.argv[1])
print("Using learning rate: {0:0.3f}".format(lr))

# Connect to Ray
if ray.is_initialized() == False:
    service_host = os.environ["RAY_HEAD_SERVICE_HOST"]
    service_port = os.environ["RAY_HEAD_SERVICE_PORT"]
    ray.init(f"ray://{service_host}:{service_port}")

# Train model
train_x, train_y = load_breast_cancer(return_X_y=True)
train_set = RayDMatrix(train_x, train_y)

evals_result = {}
bst = train(
    {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "learning_rate" : lr
    },
    train_set,
    evals_result=evals_result,
    evals=[(train_set, "train")],
    verbose_eval=False,
    ray_params=RayParams(num_actors=2, cpus_per_actor=1))

print("Final training error: {:.4f}".format(evals_result["train"]["error"][-1]))

# Save result to the Domino FS
DOMINO_USER = os.environ["DOMINO_STARTING_USERNAME"]

filename = DOMINO_USER + "_training_results.txt"

with open(filename, "w") as text_file:
    text_file.write("Learning rate used: {:.4f}\n".format(lr))
    text_file.write("Final training error: {:.4f}".format(evals_result["train"]["error"][-1]))


# Send results to user via email
# Need to configure emails & SMTP

"""

DOMINO_USER_API_KEY = os.environ["DOMINO_USER_API_KEY"]
DOMINO_PROJECT_NAME = os.environ["DOMINO_PROJECT_NAME"]
DOMINO_PROJECT_OWNER = os.environ["DOMINO_PROJECT_OWNER"]

domino_api = Domino(project=DOMINO_PROJECT_OWNER + "/" + DOMINO_PROJECT_NAME)
domino_api.authenticate(api_key=DOMINO_USER_API_KEY)

url = domino_api._routes.host + "/v4/users/self"
email = domino_api.request_manager.get(url).json()["email"].lower()

msg = MIMEText("Final training error: {:.4f}".format(evals_result["train"]["error"][-1]))

msg["Subject"] = "Your model's final training error"
msg["From"] = "..."   # <--- set to an email address of the hackathon organiser
msg["To"] = email

# Send the message via our own SMTP server, but don't include the
# envelope header.
s = smtplib.SMTP(".....") # <--- set to a local SMTP server that can deliver the email
s.sendmail(me, [you], msg.as_string())
s.quit()
"""


