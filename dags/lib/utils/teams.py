"""Generic MS Teams alerting (webhook hidden behind a secret Airflow connection)."""
import http
import logging
from urllib.parse import urlencode

import requests
from airflow.hooks.base import BaseHook

WEBHOOK_CONN_ID = 'msteams_webhook'


def get_webhook_url(conn_id=WEBHOOK_CONN_ID):
    """Reconstruct a webhook URL from an Airflow connection.

    The secret lives in the connection (provisioned via env), never in code, and is masked by
    Airflow in task logs.
    """
    conn = BaseHook.get_connection(conn_id)
    scheme = conn.conn_type or 'https'
    netloc = conn.host
    if conn.port:
        netloc = f'{netloc}:{conn.port}'
    url = f'{scheme}://{netloc}/{conn.schema or ""}'
    extra = conn.extra_dejson
    if extra:
        url = f'{url}?{urlencode(extra)}'
    return url


def send_teams_alert(context):
    """``on_failure_callback`` that posts the failed task to MS Teams."""
    webhook_url = get_webhook_url()
    dag_id = context['task_instance'].dag_id
    task_id = context['task_instance'].task_id

    message = {
        'type': 'message',
        'attachments': [
            {
                'contentType': 'application/vnd.microsoft.card.adaptive',
                'content': {
                    '$schema': 'http://adaptivecards.io/schemas/adaptive-card.json',
                    'type': 'AdaptiveCard',
                    'version': '1.4',
                    'body': [
                        {
                            'type': 'TextBlock',
                            'text': '🚨 **Airflow Task Failed!**',
                            'wrap': True,
                            'weight': 'Bolder',
                            'color': 'Attention',
                            'size': 'Medium'
                        },
                        {
                            'type': 'FactSet',
                            'facts': [
                                {'title': 'DAG', 'value': dag_id},
                                {'title': 'Task', 'value': task_id},
                            ]
                        }
                    ]
                }
            }
        ]
    }
    response = requests.post(webhook_url, json=message, timeout=30)

    if response.status_code == 200:
        logging.info('Alert send successfully')
    else:
        logging.error(f'Failed to send message to Teams: {response.status_code} {response.text}')
