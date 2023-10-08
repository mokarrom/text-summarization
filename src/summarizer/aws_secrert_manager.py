import boto3
import base64
import json
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def get_openai_api_key_from_sm():
    """Retrieve OpenAI API key from AWS Secret Manager."""

    region_name = "eu-central-1"
    secret_string = '{"openai-api-key": "None"}'
    secret_name = "chapter-summarization/api-keys/openai"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)

    try:
        secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            logger.error("The requested secret " + secret_name + " was not found")
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            logger.error("The request was invalid due to:", e)
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            logger.error("The request had invalid params:", e)
        else:
            logger.error("Error while retrieving apipass:", e)
    else:
        if 'SecretString' in secret_value_response:
            secret_string = secret_value_response['SecretString']
        else:
            secret_string = base64.b64decode(secret_value_response['SecretBinary'])
        logger.info("OpenAI API key retrieved from Secret Manager successfully.")

    return json.loads(secret_string).get("openai-api-key")


if __name__ == '__main__':
    print(f"OpenAI API Key: {get_openai_api_key_from_sm()}")

