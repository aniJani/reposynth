"""V2 benchmark: 80 tasks across httpx, Flask, FastAPI, Requests.

Each task follows V6's schema:
    id, repo, difficulty, question, relevant_file, verify, ground_truth

verify(answer: str) -> bool  judges correctness via case-insensitive
substring matching against expected facts. Kept lenient so small phrasing
variations don't flip the label, but strict on the load-bearing fact.

EASY tasks: API usage that the model probably knows from pretraining.
HARD tasks: specific defaults / class hierarchies / signatures that
require reading the source code to answer correctly.

20 per repo:
  - httpx:    lifted verbatim from research/paper/v6_tasks.py
  - Flask, FastAPI, Requests: 10 easy + 10 hard each, hand-curated.
"""

from __future__ import annotations

from research.paper.v6_tasks import TASKS as _HTTPX_TASKS_V6


# ---- httpx (V6's 20 tasks, with `repo` field added) -----------------

HTTPX = [{**t, "repo": "httpx"} for t in _HTTPX_TASKS_V6]


# ---- Flask -----------------------------------------------------------

FLASK = [
    # ===== EASY =====
    {
        "id": 101, "repo": "flask", "difficulty": "easy",
        "question": "How do you define a basic GET route in Flask?",
        "relevant_file": "src/flask/app.py",
        "verify": lambda r: "@app.route" in r.lower() or "app.route(" in r.lower(),
        "ground_truth": "@app.route('/path')",
    },
    {
        "id": 102, "repo": "flask", "difficulty": "easy",
        "question": "How do you render an HTML template in Flask?",
        "relevant_file": "src/flask/templating.py",
        "verify": lambda r: "render_template" in r.lower(),
        "ground_truth": "render_template('name.html')",
    },
    {
        "id": 103, "repo": "flask", "difficulty": "easy",
        "question": "How do you access form data submitted via POST in Flask?",
        "relevant_file": "src/flask/wrappers.py",
        "verify": lambda r: "request.form" in r.lower(),
        "ground_truth": "request.form['key']",
    },
    {
        "id": 104, "repo": "flask", "difficulty": "easy",
        "question": "How do you access URL query parameters in Flask?",
        "relevant_file": "src/flask/wrappers.py",
        "verify": lambda r: "request.args" in r.lower(),
        "ground_truth": "request.args.get('name')",
    },
    {
        "id": 105, "repo": "flask", "difficulty": "easy",
        "question": "How do you access the JSON request body in Flask?",
        "relevant_file": "src/flask/wrappers.py",
        "verify": lambda r: "request.json" in r.lower() or "get_json" in r.lower(),
        "ground_truth": "request.get_json()",
    },
    {
        "id": 106, "repo": "flask", "difficulty": "easy",
        "question": "How do you redirect to another URL in Flask?",
        "relevant_file": "src/flask/helpers.py",
        "verify": lambda r: "redirect(" in r.lower(),
        "ground_truth": "redirect(url_for('endpoint'))",
    },
    {
        "id": 107, "repo": "flask", "difficulty": "easy",
        "question": "How do you store a value in a Flask user session?",
        "relevant_file": "src/flask/sessions.py",
        "verify": lambda r: "session[" in r.lower() or "session." in r.lower(),
        "ground_truth": "session['key'] = value",
    },
    {
        "id": 108, "repo": "flask", "difficulty": "easy",
        "question": "How do you return a JSON response in Flask?",
        "relevant_file": "src/flask/json/__init__.py",
        "verify": lambda r: "jsonify" in r.lower(),
        "ground_truth": "return jsonify(data)",
    },
    {
        "id": 109, "repo": "flask", "difficulty": "easy",
        "question": "How do you abort a Flask request with a 404 error?",
        "relevant_file": "src/flask/helpers.py",
        "verify": lambda r: "abort(" in r.lower() and "404" in r,
        "ground_truth": "abort(404)",
    },
    {
        "id": 110, "repo": "flask", "difficulty": "easy",
        "question": "How do you build a URL for a named endpoint in Flask?",
        "relevant_file": "src/flask/helpers.py",
        "verify": lambda r: "url_for" in r.lower(),
        "ground_truth": "url_for('endpoint_name')",
    },

    # ===== HARD =====
    {
        "id": 111, "repo": "flask", "difficulty": "hard",
        "question": "What is the default value of the SECRET_KEY config in Flask?",
        "relevant_file": "src/flask/app.py",
        "verify": lambda r: "none" in r.lower(),
        "ground_truth": "None",
    },
    {
        "id": 112, "repo": "flask", "difficulty": "hard",
        "question": "What is the default value of PERMANENT_SESSION_LIFETIME in Flask, in days?",
        "relevant_file": "src/flask/app.py",
        "verify": lambda r: "31" in r,
        "ground_truth": "31 days",
    },
    {
        "id": 113, "repo": "flask", "difficulty": "hard",
        "question": "What is the parent class of the Flask class in flask.app?",
        "relevant_file": "src/flask/app.py",
        "verify": lambda r: "scaffold" in r.lower() or "app(" in r.lower(),
        "ground_truth": "Scaffold (or App in newer versions)",
    },
    {
        "id": 114, "repo": "flask", "difficulty": "hard",
        "question": "What is the WSGI dispatch method on the Flask class called?",
        "relevant_file": "src/flask/app.py",
        "verify": lambda r: "wsgi_app" in r.lower(),
        "ground_truth": "wsgi_app",
    },
    {
        "id": 115, "repo": "flask", "difficulty": "hard",
        "question": "What is the default JSON provider class used by Flask 2.2+?",
        "relevant_file": "src/flask/json/provider.py",
        "verify": lambda r: "defaultjsonprovider" in r.lower(),
        "ground_truth": "DefaultJSONProvider",
    },
    {
        "id": 116, "repo": "flask", "difficulty": "hard",
        "question": "Which Flask signal fires when a request finishes successfully?",
        "relevant_file": "src/flask/signals.py",
        "verify": lambda r: "request_finished" in r.lower(),
        "ground_truth": "request_finished",
    },
    {
        "id": 117, "repo": "flask", "difficulty": "hard",
        "question": "What attribute holds the Werkzeug Map of all routes on a Flask app?",
        "relevant_file": "src/flask/app.py",
        "verify": lambda r: "url_map" in r.lower(),
        "ground_truth": "url_map",
    },
    {
        "id": 118, "repo": "flask", "difficulty": "hard",
        "question": "What is the name of the global object Flask provides for sharing data within a request, accessed as 'g'?",
        "relevant_file": "src/flask/ctx.py",
        "verify": lambda r: "appctxglobals" in r.lower() or "_appctxglobals" in r.lower(),
        "ground_truth": "_AppCtxGlobals",
    },
    {
        "id": 119, "repo": "flask", "difficulty": "hard",
        "question": "What is the default value of MAX_CONTENT_LENGTH in Flask?",
        "relevant_file": "src/flask/app.py",
        "verify": lambda r: "none" in r.lower(),
        "ground_truth": "None",
    },
    {
        "id": 120, "repo": "flask", "difficulty": "hard",
        "question": "What is the default value of the JSON_SORT_KEYS config option in Flask?",
        "relevant_file": "src/flask/json/provider.py",
        "verify": lambda r: "true" in r.lower(),
        "ground_truth": "True",
    },
]


# ---- FastAPI ---------------------------------------------------------

FASTAPI = [
    # ===== EASY =====
    {
        "id": 201, "repo": "fastapi", "difficulty": "easy",
        "question": "How do you define a GET endpoint in FastAPI?",
        "relevant_file": "fastapi/applications.py",
        "verify": lambda r: "@app.get" in r.lower() or "app.get(" in r.lower(),
        "ground_truth": "@app.get('/path')",
    },
    {
        "id": 202, "repo": "fastapi", "difficulty": "easy",
        "question": "How do you define a POST endpoint that accepts a JSON body in FastAPI?",
        "relevant_file": "fastapi/routing.py",
        "verify": lambda r: ("@app.post" in r.lower() or "app.post(" in r.lower()) and "basemodel" in r.lower(),
        "ground_truth": "@app.post('/items'); def create(item: ItemModel(BaseModel))",
    },
    {
        "id": 203, "repo": "fastapi", "difficulty": "easy",
        "question": "How do you declare a path parameter in a FastAPI route?",
        "relevant_file": "fastapi/routing.py",
        "verify": lambda r: "{" in r and "}" in r and ("item_id" in r.lower() or "/{" in r.lower()),
        "ground_truth": "@app.get('/items/{item_id}')",
    },
    {
        "id": 204, "repo": "fastapi", "difficulty": "easy",
        "question": "How do you declare a query parameter with a default value in FastAPI?",
        "relevant_file": "fastapi/params.py",
        "verify": lambda r: "=" in r and ("query" in r.lower() or "def " in r.lower()),
        "ground_truth": "def read(skip: int = 0)",
    },
    {
        "id": 205, "repo": "fastapi", "difficulty": "easy",
        "question": "How do you use Pydantic to validate a request body in FastAPI?",
        "relevant_file": "fastapi/routing.py",
        "verify": lambda r: "basemodel" in r.lower(),
        "ground_truth": "class Item(BaseModel): ...; def f(item: Item)",
    },
    {
        "id": 206, "repo": "fastapi", "difficulty": "easy",
        "question": "How do you declare a dependency in FastAPI?",
        "relevant_file": "fastapi/dependencies/utils.py",
        "verify": lambda r: "depends" in r.lower(),
        "ground_truth": "Depends(some_callable)",
    },
    {
        "id": 207, "repo": "fastapi", "difficulty": "easy",
        "question": "How do you raise an HTTP 404 from inside a FastAPI route?",
        "relevant_file": "fastapi/exceptions.py",
        "verify": lambda r: "httpexception" in r.lower() and "404" in r,
        "ground_truth": "raise HTTPException(status_code=404)",
    },
    {
        "id": 208, "repo": "fastapi", "difficulty": "easy",
        "question": "How do you declare an async endpoint in FastAPI?",
        "relevant_file": "fastapi/routing.py",
        "verify": lambda r: "async def" in r.lower(),
        "ground_truth": "async def endpoint(): ...",
    },
    {
        "id": 209, "repo": "fastapi", "difficulty": "easy",
        "question": "How do you set a custom response status code on a FastAPI route?",
        "relevant_file": "fastapi/routing.py",
        "verify": lambda r: "status_code" in r.lower(),
        "ground_truth": "@app.post('/x', status_code=201)",
    },
    {
        "id": 210, "repo": "fastapi", "difficulty": "easy",
        "question": "How do you run background tasks after returning a response in FastAPI?",
        "relevant_file": "fastapi/background.py",
        "verify": lambda r: "backgroundtasks" in r.lower(),
        "ground_truth": "BackgroundTasks (DI parameter, .add_task)",
    },

    # ===== HARD =====
    {
        "id": 211, "repo": "fastapi", "difficulty": "hard",
        "question": "What is the parent class of the FastAPI application class?",
        "relevant_file": "fastapi/applications.py",
        "verify": lambda r: "starlette" in r.lower(),
        "ground_truth": "Starlette",
    },
    {
        "id": 212, "repo": "fastapi", "difficulty": "hard",
        "question": "What is the default value of the docs_url parameter in FastAPI()?",
        "relevant_file": "fastapi/applications.py",
        "verify": lambda r: "/docs" in r.lower(),
        "ground_truth": "/docs",
    },
    {
        "id": 213, "repo": "fastapi", "difficulty": "hard",
        "question": "What is the default value of the redoc_url parameter in FastAPI()?",
        "relevant_file": "fastapi/applications.py",
        "verify": lambda r: "/redoc" in r.lower(),
        "ground_truth": "/redoc",
    },
    {
        "id": 214, "repo": "fastapi", "difficulty": "hard",
        "question": "What is the parent class of HTTPException in fastapi.exceptions?",
        "relevant_file": "fastapi/exceptions.py",
        "verify": lambda r: "starletthttpexception" in r.lower() or "starlette" in r.lower(),
        "ground_truth": "starlette.exceptions.HTTPException",
    },
    {
        "id": 215, "repo": "fastapi", "difficulty": "hard",
        "question": "What is the default value of the openapi_url parameter in FastAPI()?",
        "relevant_file": "fastapi/applications.py",
        "verify": lambda r: "/openapi.json" in r.lower(),
        "ground_truth": "/openapi.json",
    },
    {
        "id": 216, "repo": "fastapi", "difficulty": "hard",
        "question": "Which class does FastAPI's APIRouter inherit from?",
        "relevant_file": "fastapi/routing.py",
        "verify": lambda r: "router" in r.lower() and "starlette" in r.lower(),
        "ground_truth": "starlette.routing.Router",
    },
    {
        "id": 217, "repo": "fastapi", "difficulty": "hard",
        "question": "What is the name of FastAPI's parameter wrapper class for query parameters?",
        "relevant_file": "fastapi/params.py",
        "verify": lambda r: "query" in r.lower() and "class" in r.lower(),
        "ground_truth": "Query",
    },
    {
        "id": 218, "repo": "fastapi", "difficulty": "hard",
        "question": "Which exception does FastAPI raise when Pydantic validation of a request fails?",
        "relevant_file": "fastapi/exceptions.py",
        "verify": lambda r: "requestvalidationerror" in r.lower(),
        "ground_truth": "RequestValidationError",
    },
    {
        "id": 219, "repo": "fastapi", "difficulty": "hard",
        "question": "What is the default value of include_in_schema in FastAPI's @app.get?",
        "relevant_file": "fastapi/routing.py",
        "verify": lambda r: "true" in r.lower(),
        "ground_truth": "True",
    },
    {
        "id": 220, "repo": "fastapi", "difficulty": "hard",
        "question": "What class does FastAPI use as its default JSON response class?",
        "relevant_file": "fastapi/responses.py",
        "verify": lambda r: "jsonresponse" in r.lower(),
        "ground_truth": "JSONResponse",
    },
]


# ---- Requests --------------------------------------------------------

REQUESTS = [
    # ===== EASY =====
    {
        "id": 301, "repo": "requests", "difficulty": "easy",
        "question": "How do you make a simple GET request using the requests library?",
        "relevant_file": "src/requests/api.py",
        "verify": lambda r: "requests.get" in r.lower(),
        "ground_truth": "requests.get(url)",
    },
    {
        "id": 302, "repo": "requests", "difficulty": "easy",
        "question": "How do you POST JSON data with the requests library?",
        "relevant_file": "src/requests/api.py",
        "verify": lambda r: "requests.post" in r.lower() and "json" in r.lower(),
        "ground_truth": "requests.post(url, json=data)",
    },
    {
        "id": 303, "repo": "requests", "difficulty": "easy",
        "question": "How do you send a GET request with query parameters using requests?",
        "relevant_file": "src/requests/api.py",
        "verify": lambda r: "params" in r.lower(),
        "ground_truth": "requests.get(url, params={'q': 'x'})",
    },
    {
        "id": 304, "repo": "requests", "difficulty": "easy",
        "question": "How do you set custom headers on a requests call?",
        "relevant_file": "src/requests/api.py",
        "verify": lambda r: "headers" in r.lower(),
        "ground_truth": "requests.get(url, headers={'X-X':'y'})",
    },
    {
        "id": 305, "repo": "requests", "difficulty": "easy",
        "question": "How do you access the response body as text in requests?",
        "relevant_file": "src/requests/models.py",
        "verify": lambda r: ".text" in r.lower(),
        "ground_truth": "response.text",
    },
    {
        "id": 306, "repo": "requests", "difficulty": "easy",
        "question": "How do you parse a JSON response body in requests?",
        "relevant_file": "src/requests/models.py",
        "verify": lambda r: ".json()" in r.lower(),
        "ground_truth": "response.json()",
    },
    {
        "id": 307, "repo": "requests", "difficulty": "easy",
        "question": "How do you check the status code of a response in requests?",
        "relevant_file": "src/requests/models.py",
        "verify": lambda r: "status_code" in r.lower(),
        "ground_truth": "response.status_code",
    },
    {
        "id": 308, "repo": "requests", "difficulty": "easy",
        "question": "How do you create a persistent Session in requests?",
        "relevant_file": "src/requests/sessions.py",
        "verify": lambda r: "session" in r.lower() and ("with" in r.lower() or "session()" in r.lower()),
        "ground_truth": "with requests.Session() as s:",
    },
    {
        "id": 309, "repo": "requests", "difficulty": "easy",
        "question": "How do you set a request timeout in requests?",
        "relevant_file": "src/requests/api.py",
        "verify": lambda r: "timeout" in r.lower(),
        "ground_truth": "requests.get(url, timeout=10)",
    },
    {
        "id": 310, "repo": "requests", "difficulty": "easy",
        "question": "How do you send Basic auth credentials with a requests call?",
        "relevant_file": "src/requests/auth.py",
        "verify": lambda r: "auth=" in r.lower() or "httpbasicauth" in r.lower(),
        "ground_truth": "requests.get(url, auth=(user, pwd))",
    },

    # ===== HARD =====
    {
        "id": 311, "repo": "requests", "difficulty": "hard",
        "question": "What is the default value of allow_redirects for a requests.get call?",
        "relevant_file": "src/requests/api.py",
        "verify": lambda r: "true" in r.lower(),
        "ground_truth": "True",
    },
    {
        "id": 312, "repo": "requests", "difficulty": "hard",
        "question": "What is the default value of verify (SSL cert verification) in requests?",
        "relevant_file": "src/requests/sessions.py",
        "verify": lambda r: "true" in r.lower(),
        "ground_truth": "True",
    },
    {
        "id": 313, "repo": "requests", "difficulty": "hard",
        "question": "What is the parent class of HTTPError in requests.exceptions?",
        "relevant_file": "src/requests/exceptions.py",
        "verify": lambda r: "requestexception" in r.lower(),
        "ground_truth": "RequestException",
    },
    {
        "id": 314, "repo": "requests", "difficulty": "hard",
        "question": "What is the parent class of RequestException in requests.exceptions?",
        "relevant_file": "src/requests/exceptions.py",
        "verify": lambda r: "ioerror" in r.lower() or "oserror" in r.lower(),
        "ground_truth": "IOError",
    },
    {
        "id": 315, "repo": "requests", "difficulty": "hard",
        "question": "What is the default DEFAULT_REDIRECT_LIMIT in requests.models?",
        "relevant_file": "src/requests/models.py",
        "verify": lambda r: "30" in r,
        "ground_truth": "30",
    },
    {
        "id": 316, "repo": "requests", "difficulty": "hard",
        "question": "Name three authentication classes provided by requests.auth.",
        "relevant_file": "src/requests/auth.py",
        "verify": lambda r: "httpbasicauth" in r.lower() and ("httpdigestauth" in r.lower() or "httpproxyauth" in r.lower()),
        "ground_truth": "HTTPBasicAuth, HTTPDigestAuth, HTTPProxyAuth",
    },
    {
        "id": 317, "repo": "requests", "difficulty": "hard",
        "question": "Which exception does requests raise when an SSL certificate cannot be verified?",
        "relevant_file": "src/requests/exceptions.py",
        "verify": lambda r: "sslerror" in r.lower(),
        "ground_truth": "SSLError",
    },
    {
        "id": 318, "repo": "requests", "difficulty": "hard",
        "question": "Which exception does requests raise when a request times out?",
        "relevant_file": "src/requests/exceptions.py",
        "verify": lambda r: "timeout" in r.lower(),
        "ground_truth": "Timeout (or ConnectTimeout / ReadTimeout)",
    },
    {
        "id": 319, "repo": "requests", "difficulty": "hard",
        "question": "What is the default value of stream in a requests.get call?",
        "relevant_file": "src/requests/api.py",
        "verify": lambda r: "false" in r.lower(),
        "ground_truth": "False",
    },
    {
        "id": 320, "repo": "requests", "difficulty": "hard",
        "question": "What is the default User-Agent string format used by requests?",
        "relevant_file": "src/requests/utils.py",
        "verify": lambda r: "python-requests" in r.lower(),
        "ground_truth": "python-requests/<version>",
    },
]


# ---- combined --------------------------------------------------------

TASKS = HTTPX + FLASK + FASTAPI + REQUESTS


# Repo metadata: how to clone, where source files live, what file-path
# prefix the tasks' `relevant_file` values use.
REPOS = {
    "httpx":    {"clone_url": "https://github.com/encode/httpx.git",
                  "source_root": "httpx",
                  "depth": 1},
    "flask":    {"clone_url": "https://github.com/pallets/flask.git",
                  "source_root": "src/flask",
                  "depth": 1},
    "fastapi":  {"clone_url": "https://github.com/fastapi/fastapi.git",
                  "source_root": "fastapi",
                  "depth": 1},
    "requests": {"clone_url": "https://github.com/psf/requests.git",
                  "source_root": "src/requests",
                  "depth": 1},
}


if __name__ == "__main__":
    print(f"v2 benchmark: {len(TASKS)} total tasks")
    for repo in REPOS:
        n = sum(1 for t in TASKS if t["repo"] == repo)
        e = sum(1 for t in TASKS if t["repo"] == repo and t["difficulty"] == "easy")
        h = sum(1 for t in TASKS if t["repo"] == repo and t["difficulty"] == "hard")
        print(f"  {repo:9s}: {n} ({e} easy, {h} hard)")
