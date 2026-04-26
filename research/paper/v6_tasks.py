"""V6's 20 balanced tasks, lifted verbatim from Week15_V6_Balanced_Tasks.ipynb cell 6.

Verify functions are kept as lambdas so the PoC can run without re-deriving them.
Used by research/paper/cce_poc.py.
"""

from __future__ import annotations


TASKS = [
    # ===== EASY TASKS (model likely knows from training) =====
    {
        'id': 1, 'difficulty': 'easy',
        'question': 'How do you make a simple GET request using httpx?',
        'relevant_file': 'httpx/_api.py',
        'verify': lambda r: 'httpx.get' in r.lower() or 'get(' in r.lower(),
        'ground_truth': 'httpx.get(url)',
    },
    {
        'id': 2, 'difficulty': 'easy',
        'question': 'How do you make a POST request with JSON data using httpx?',
        'relevant_file': 'httpx/_api.py',
        'verify': lambda r: 'post' in r.lower() and 'json' in r.lower(),
        'ground_truth': 'httpx.post(url, json=data)',
    },
    {
        'id': 3, 'difficulty': 'easy',
        'question': 'How do you access the response body as text in httpx?',
        'relevant_file': 'httpx/_models.py',
        'verify': lambda r: '.text' in r.lower() or 'response.text' in r.lower(),
        'ground_truth': 'response.text',
    },
    {
        'id': 4, 'difficulty': 'easy',
        'question': 'How do you access the JSON response body in httpx?',
        'relevant_file': 'httpx/_models.py',
        'verify': lambda r: '.json()' in r.lower() or 'response.json' in r.lower(),
        'ground_truth': 'response.json()',
    },
    {
        'id': 5, 'difficulty': 'easy',
        'question': 'How do you check the HTTP status code of a response in httpx?',
        'relevant_file': 'httpx/_models.py',
        'verify': lambda r: 'status_code' in r.lower(),
        'ground_truth': 'response.status_code',
    },
    {
        'id': 6, 'difficulty': 'easy',
        'question': 'How do you create an httpx Client for making multiple requests?',
        'relevant_file': 'httpx/_client.py',
        'verify': lambda r: 'client' in r.lower() and ('with' in r.lower() or 'httpx.client' in r.lower()),
        'ground_truth': 'with httpx.Client() as client:',
    },
    {
        'id': 7, 'difficulty': 'easy',
        'question': 'How do you set a timeout for requests in httpx?',
        'relevant_file': 'httpx/_config.py',
        'verify': lambda r: 'timeout' in r.lower(),
        'ground_truth': 'httpx.get(url, timeout=10.0)',
    },
    {
        'id': 8, 'difficulty': 'easy',
        'question': 'How do you add custom headers to an httpx request?',
        'relevant_file': 'httpx/_api.py',
        'verify': lambda r: 'headers' in r.lower(),
        'ground_truth': 'httpx.get(url, headers={"X-Custom": "value"})',
    },
    {
        'id': 9, 'difficulty': 'easy',
        'question': 'Does httpx support async requests? If so, how?',
        'relevant_file': 'httpx/_client.py',
        'verify': lambda r: 'async' in r.lower() and ('asyncclient' in r.lower() or 'await' in r.lower()),
        'ground_truth': 'async with httpx.AsyncClient() as client: await client.get(url)',
    },
    {
        'id': 10, 'difficulty': 'easy',
        'question': 'How do you send form data in an httpx POST request?',
        'relevant_file': 'httpx/_api.py',
        'verify': lambda r: 'data' in r.lower() or 'form' in r.lower(),
        'ground_truth': 'httpx.post(url, data={"key": "value"})',
    },

    # ===== HARD TASKS (require source code) =====
    {
        'id': 11, 'difficulty': 'hard',
        'question': 'What is the exact default timeout value in seconds for httpx requests?',
        'relevant_file': 'httpx/_config.py',
        'verify': lambda r: '5.0' in r or ('5' in r and '30' not in r and '10' not in r),
        'ground_truth': '5.0 seconds',
    },
    {
        'id': 12, 'difficulty': 'hard',
        'question': 'What is the exact value of DEFAULT_MAX_REDIRECTS in httpx?',
        'relevant_file': 'httpx/_config.py',
        'verify': lambda r: '20' in r,
        'ground_truth': '20',
    },
    {
        'id': 13, 'difficulty': 'hard',
        'question': 'What are the exact 3 parameters of httpx.Limits.__init__()?',
        'relevant_file': 'httpx/_config.py',
        'verify': lambda r: 'max_connections' in r.lower() and 'keepalive_expiry' in r.lower() and 'max_retries' not in r.lower(),
        'ground_truth': 'max_connections, max_keepalive_connections, keepalive_expiry',
    },
    {
        'id': 14, 'difficulty': 'hard',
        'question': 'What is the default value for max_connections in httpx.Limits? Is it 100, 50, or None?',
        'relevant_file': 'httpx/_config.py',
        'verify': lambda r: 'none' in r.lower(),
        'ground_truth': 'None',
    },
    {
        'id': 15, 'difficulty': 'hard',
        'question': 'List ALL 5 authentication classes defined in httpx._auth module.',
        'relevant_file': 'httpx/_auth.py',
        'verify': lambda r: 'basicauth' in r.lower() and 'digestauth' in r.lower() and 'netrcauth' in r.lower(),
        'ground_truth': 'Auth, BasicAuth, DigestAuth, FunctionAuth, NetRCAuth',
    },
    {
        'id': 16, 'difficulty': 'hard',
        'question': 'What is the parent class of ConnectError in httpx._exceptions? Is it Exception, HTTPError, or NetworkError?',
        'relevant_file': 'httpx/_exceptions.py',
        'verify': lambda r: 'networkerror' in r.lower(),
        'ground_truth': 'NetworkError',
    },
    {
        'id': 17, 'difficulty': 'hard',
        'question': 'What exception does httpx raise for unsupported URL schemes like ftp://?',
        'relevant_file': 'httpx/_exceptions.py',
        'verify': lambda r: 'unsupportedprotocol' in r.lower(),
        'ground_truth': 'UnsupportedProtocol',
    },
    {
        'id': 18, 'difficulty': 'hard',
        'question': 'What are the 2 base transport classes in httpx._transports.base?',
        'relevant_file': 'httpx/_transports/base.py',
        'verify': lambda r: 'basetransport' in r.lower() and 'asyncbasetransport' in r.lower(),
        'ground_truth': 'BaseTransport, AsyncBaseTransport',
    },
    {
        'id': 19, 'difficulty': 'hard',
        'question': 'What is the default value of keepalive_expiry in httpx.Limits?',
        'relevant_file': 'httpx/_config.py',
        'verify': lambda r: '5.0' in r or '5 ' in r or 'five' in r.lower(),
        'ground_truth': '5.0 seconds',
    },
    {
        'id': 20, 'difficulty': 'hard',
        'question': 'What are the 5 timeout components in httpx.Timeout (timeout, connect, read, write, and what else)?',
        'relevant_file': 'httpx/_config.py',
        'verify': lambda r: 'pool' in r.lower(),
        'ground_truth': 'timeout, connect, read, write, pool',
    },
]
