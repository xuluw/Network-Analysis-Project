import sys
import requests
from bs4 import BeautifulSoup

SEED_URL = 'https://www.linkedin.com/uas/login'
LOGIN_URL = 'https://www.linkedin.com/checkpoint/lg/login-submit'
VERIFY_URL = 'https://www.linkedin.com/checkpoint/challenge/verify'

session = requests.Session()


def login(email, password):
    session.get(SEED_URL)
    text = session.get(SEED_URL).text
    soup = BeautifulSoup(text, 'html.parser')
    payload = {'session_key': email,
               'loginCsrfParam': soup.find('input', {'name': 'loginCsrfParam'})['value'],
               'session_password': password}

    r = session.post(LOGIN_URL, data=payload)
    soup = BeautifulSoup(r.text, 'html.parser')
    verify_pin(soup)


def verify_pin(soup):
    pin = input('Check the PIN in your inbox and enter here:\n')
    payload = {
        'csrfToken': soup.find('input', {'name': 'csrfToken'})['value'],
        'pageInstance': soup.find('input', {'name': 'pageInstance'})['value'],
        'resendUrl': soup.find('input', {'name': 'resendUrl'})['value'],
        'challengeId': soup.find('input', {'name': 'challengeId'})['value'],
        'language': 'en-US',
        'displayTime': soup.find('input', {'name': 'displayTime'})['value'],
        'challengeSource': soup.find('input', {'name': 'challengeSource'})['value'],
        'requestSubmissionId': soup.find('input', {'name': 'requestSubmissionId'})['value'],
        'challengeType': soup.find('input', {'name': 'challengeType'})['value'],
        'challengeData': soup.find('input', {'name': 'challengeData'})['value'],
        'challengeDetails': soup.find('input', {'name': 'challengeDetails'})['value'],
        'failureRedirectUri': soup.find('input', {'name': 'failureRedirectUri'})['value'],
        'pin': pin
    }
    session.post(VERIFY_URL, data=payload)


if __name__ == '__main__':
    email = 'xw370@georgetown.edu' #sys.argv[1]
    password = 'yuzuru~1015' #sys.argv[2]
    login(email, password)