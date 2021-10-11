
import sys

users = {
    "nino": ['', '', ''],
    "daniel": ['/Users/layetri/Development/csd3/source', '/Users/layetri/opt/anaconda3/lib/python3.8', '/Users/layetri/opt/anaconda3/lib/python3.8/site-packages'],
    "wouter": ['/Users/wouter/Documents/School/Jaar_3/CSD/csd3/source', '/Users/wouter/anaconda3/envs/source/lib/python3.7', '/Users/wouter/anaconda3/envs/source/lib/python3.7/site-packages'],
    "paul": ['', '', ''],
}

user = "daniel"

sys.path.append(users[user][0])
sys.path.append(users[user][1])
sys.path.append(users[user][2])