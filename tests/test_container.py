import json
import sys
import time
import requests
import subprocess
from pytest import fixture

BASE_URL = "http://localhost:8080/"
PING_URL = BASE_URL + "ping"
INVOCATION_URL = BASE_URL + "invocations"
CONTAINER_NAME = "chapter-sum"


@fixture(scope="session")
def image_uri(pytestconfig):
    return pytestconfig.getoption("image_uri")


@fixture(scope="module", autouse=True)
def container(image_uri):
    """Runs a container for integration testing, removes it automatically after the testing is completed."""
    command = f"docker run --name {CONTAINER_NAME} \
        -p 8080:8080 {image_uri} serve"
    try:
        proc = subprocess.Popen(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)

        attempts = 0
        while attempts < 10:
            time.sleep(3)
            try:
                requests.get(PING_URL)
                break
            except Exception:
                attempts += 1
                pass
        yield proc.pid

    finally:
        command = f"docker rm -f {CONTAINER_NAME}"
        subprocess.check_call(command.split())


def test_ping():
    """Tests the availability of the url."""
    res = requests.get(PING_URL)
    assert res.status_code == 200


def make_invocation_request(data, content_type="application/json"):
    headers = {"Content-Type": content_type}
    response = requests.post(INVOCATION_URL, data=data, headers=headers)
    return response.status_code, response.content


def test_invocation(sample_file):
    """Tests the deployed model's output by hitting the endpoint."""
    with open(sample_file, "r") as file_obj:
        long_text = file_obj.read()
        payload = {
            "text": long_text
        }

    status_code, resp_data = make_invocation_request(json.dumps(payload))
    assert status_code == 200, f"Error message: {resp_data}"
    resp_dict = json.loads(resp_data)
    assert "summary" in resp_dict
    print(resp_dict["summary"])


