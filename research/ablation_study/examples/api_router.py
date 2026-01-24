"""
API Router Module

Provides HTTP routing, middleware support, and request handling
for building RESTful APIs.
"""

import re
import json
from typing import Callable, Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import traceback


class HTTPMethod(Enum):
    """HTTP methods supported by the router."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


@dataclass
class Request:
    """Represents an HTTP request."""
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    path_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    json_body: Optional[Dict] = None

    @property
    def content_type(self) -> Optional[str]:
        return self.headers.get('Content-Type', self.headers.get('content-type'))

    @property
    def authorization(self) -> Optional[str]:
        return self.headers.get('Authorization', self.headers.get('authorization'))

    def get_json(self) -> Optional[Dict]:
        """Parse and return JSON body."""
        if self.json_body is not None:
            return self.json_body
        if self.body:
            try:
                self.json_body = json.loads(self.body.decode('utf-8'))
                return self.json_body
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
        return None


@dataclass
class Response:
    """Represents an HTTP response."""
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None

    @classmethod
    def json(cls, data: Any, status_code: int = 200) -> 'Response':
        """Create a JSON response."""
        body = json.dumps(data).encode('utf-8')
        return cls(
            status_code=status_code,
            headers={'Content-Type': 'application/json'},
            body=body
        )

    @classmethod
    def text(cls, text: str, status_code: int = 200) -> 'Response':
        """Create a plain text response."""
        return cls(
            status_code=status_code,
            headers={'Content-Type': 'text/plain'},
            body=text.encode('utf-8')
        )

    @classmethod
    def html(cls, html: str, status_code: int = 200) -> 'Response':
        """Create an HTML response."""
        return cls(
            status_code=status_code,
            headers={'Content-Type': 'text/html'},
            body=html.encode('utf-8')
        )

    @classmethod
    def redirect(cls, url: str, permanent: bool = False) -> 'Response':
        """Create a redirect response."""
        return cls(
            status_code=301 if permanent else 302,
            headers={'Location': url}
        )

    @classmethod
    def error(cls, message: str, status_code: int = 500) -> 'Response':
        """Create an error response."""
        return cls.json({'error': message}, status_code)


@dataclass
class Route:
    """Represents a registered route."""
    path: str
    method: HTTPMethod
    handler: Callable
    middlewares: List[Callable] = field(default_factory=list)
    name: Optional[str] = None
    _pattern: re.Pattern = field(init=False, repr=False)
    _param_names: List[str] = field(init=False, repr=False)

    def __post_init__(self):
        # Convert path params like /users/{id} to regex
        self._param_names = re.findall(r'\{(\w+)\}', self.path)
        pattern = self.path
        for param in self._param_names:
            pattern = pattern.replace(f'{{{param}}}', r'(?P<' + param + r'>[^/]+)')
        self._pattern = re.compile(f'^{pattern}$')

    def matches(self, path: str, method: str) -> Optional[Dict[str, str]]:
        """
        Check if the route matches the given path and method.

        Returns:
            Path parameters if matched, None otherwise
        """
        if self.method.value != method:
            return None

        match = self._pattern.match(path)
        if match:
            return match.groupdict()
        return None


# Type alias for middleware functions
Middleware = Callable[[Request, Callable], Response]


class Router:
    """
    HTTP router with support for path parameters, middlewares, and sub-routers.

    Usage:
        router = Router()

        @router.get('/users')
        def list_users(request):
            return Response.json({'users': []})

        @router.get('/users/{id}')
        def get_user(request):
            user_id = request.path_params['id']
            return Response.json({'id': user_id})

        @router.post('/users')
        def create_user(request):
            data = request.get_json()
            return Response.json({'id': 1, **data}, status_code=201)
    """

    def __init__(self, prefix: str = ""):
        self.prefix = prefix.rstrip('/')
        self.routes: List[Route] = []
        self.middlewares: List[Middleware] = []
        self.sub_routers: List[Tuple[str, 'Router']] = []
        self.error_handlers: Dict[int, Callable] = {}
        self.exception_handler: Optional[Callable] = None

    def add_route(
        self,
        path: str,
        method: HTTPMethod,
        handler: Callable,
        middlewares: List[Middleware] = None,
        name: str = None
    ) -> Route:
        """Register a route."""
        full_path = self.prefix + path
        route = Route(
            path=full_path,
            method=method,
            handler=handler,
            middlewares=middlewares or [],
            name=name
        )
        self.routes.append(route)
        return route

    def get(self, path: str, middlewares: List[Middleware] = None, name: str = None):
        """Decorator for GET routes."""
        def decorator(handler: Callable) -> Callable:
            self.add_route(path, HTTPMethod.GET, handler, middlewares, name)
            return handler
        return decorator

    def post(self, path: str, middlewares: List[Middleware] = None, name: str = None):
        """Decorator for POST routes."""
        def decorator(handler: Callable) -> Callable:
            self.add_route(path, HTTPMethod.POST, handler, middlewares, name)
            return handler
        return decorator

    def put(self, path: str, middlewares: List[Middleware] = None, name: str = None):
        """Decorator for PUT routes."""
        def decorator(handler: Callable) -> Callable:
            self.add_route(path, HTTPMethod.PUT, handler, middlewares, name)
            return handler
        return decorator

    def patch(self, path: str, middlewares: List[Middleware] = None, name: str = None):
        """Decorator for PATCH routes."""
        def decorator(handler: Callable) -> Callable:
            self.add_route(path, HTTPMethod.PATCH, handler, middlewares, name)
            return handler
        return decorator

    def delete(self, path: str, middlewares: List[Middleware] = None, name: str = None):
        """Decorator for DELETE routes."""
        def decorator(handler: Callable) -> Callable:
            self.add_route(path, HTTPMethod.DELETE, handler, middlewares, name)
            return handler
        return decorator

    def use(self, middleware: Middleware) -> None:
        """Add a global middleware."""
        self.middlewares.append(middleware)

    def include_router(self, router: 'Router', prefix: str = "") -> None:
        """Include another router with an optional prefix."""
        self.sub_routers.append((prefix, router))

    def error_handler(self, status_code: int):
        """Decorator for custom error handlers."""
        def decorator(handler: Callable) -> Callable:
            self.error_handlers[status_code] = handler
            return handler
        return decorator

    def exception_handler_decorator(self, handler: Callable) -> Callable:
        """Decorator for global exception handler."""
        self.exception_handler = handler
        return handler

    def find_route(self, path: str, method: str) -> Optional[Tuple[Route, Dict[str, str]]]:
        """
        Find a matching route for the given path and method.

        Returns:
            Tuple of (route, path_params) if found, None otherwise
        """
        # Check own routes
        for route in self.routes:
            params = route.matches(path, method)
            if params is not None:
                return route, params

        # Check sub-routers
        for prefix, router in self.sub_routers:
            full_prefix = self.prefix + prefix
            if path.startswith(full_prefix) or full_prefix == "":
                result = router.find_route(path, method)
                if result:
                    return result

        return None

    def handle_request(self, request: Request) -> Response:
        """
        Handle an incoming HTTP request.

        Args:
            request: The HTTP request to handle

        Returns:
            HTTP response
        """
        try:
            # Find matching route
            result = self.find_route(request.path, request.method)

            if result is None:
                return self._handle_error(404, request)

            route, path_params = result
            request.path_params = path_params

            # Build middleware chain
            handler = route.handler
            middlewares = self.middlewares + route.middlewares

            for middleware in reversed(middlewares):
                handler = self._wrap_middleware(middleware, handler)

            # Execute handler
            response = handler(request)

            if not isinstance(response, Response):
                response = Response.json(response)

            return response

        except Exception as e:
            if self.exception_handler:
                return self.exception_handler(request, e)
            return self._handle_exception(e)

    def _wrap_middleware(
        self,
        middleware: Middleware,
        handler: Callable
    ) -> Callable:
        """Wrap a handler with a middleware."""
        def wrapped(request: Request) -> Response:
            return middleware(request, handler)
        return wrapped

    def _handle_error(self, status_code: int, request: Request) -> Response:
        """Handle HTTP errors."""
        if status_code in self.error_handlers:
            return self.error_handlers[status_code](request)

        messages = {
            400: 'Bad Request',
            401: 'Unauthorized',
            403: 'Forbidden',
            404: 'Not Found',
            405: 'Method Not Allowed',
            500: 'Internal Server Error',
        }

        return Response.error(messages.get(status_code, 'Error'), status_code)

    def _handle_exception(self, exception: Exception) -> Response:
        """Handle uncaught exceptions."""
        return Response.json({
            'error': 'Internal Server Error',
            'message': str(exception),
            'type': type(exception).__name__
        }, status_code=500)

    def get_routes(self) -> List[Dict[str, Any]]:
        """Get all registered routes for documentation."""
        routes = []

        for route in self.routes:
            routes.append({
                'path': route.path,
                'method': route.method.value,
                'name': route.name,
                'handler': route.handler.__name__
            })

        for prefix, router in self.sub_routers:
            for route_info in router.get_routes():
                route_info['path'] = prefix + route_info['path']
                routes.append(route_info)

        return routes


# Common middlewares

def cors_middleware(
    allowed_origins: List[str] = None,
    allowed_methods: List[str] = None,
    allowed_headers: List[str] = None,
    max_age: int = 86400
) -> Middleware:
    """
    CORS middleware factory.

    Usage:
        router.use(cors_middleware(
            allowed_origins=['https://example.com'],
            allowed_methods=['GET', 'POST'],
        ))
    """
    allowed_origins = allowed_origins or ['*']
    allowed_methods = allowed_methods or ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
    allowed_headers = allowed_headers or ['Content-Type', 'Authorization']

    def middleware(request: Request, next_handler: Callable) -> Response:
        origin = request.headers.get('Origin', '*')

        # Handle preflight
        if request.method == 'OPTIONS':
            return Response(
                status_code=204,
                headers={
                    'Access-Control-Allow-Origin': origin if origin in allowed_origins or '*' in allowed_origins else '',
                    'Access-Control-Allow-Methods': ', '.join(allowed_methods),
                    'Access-Control-Allow-Headers': ', '.join(allowed_headers),
                    'Access-Control-Max-Age': str(max_age),
                }
            )

        # Process request
        response = next_handler(request)

        # Add CORS headers
        if origin in allowed_origins or '*' in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin

        return response

    return middleware


def logging_middleware(logger: Callable = print) -> Middleware:
    """
    Request logging middleware.

    Usage:
        router.use(logging_middleware())
    """
    def middleware(request: Request, next_handler: Callable) -> Response:
        import time
        start = time.time()

        response = next_handler(request)

        duration = (time.time() - start) * 1000
        logger(f"{request.method} {request.path} - {response.status_code} ({duration:.2f}ms)")

        return response

    return middleware


def auth_middleware(
    auth_service,
    excluded_paths: List[str] = None
) -> Middleware:
    """
    Authentication middleware.

    Usage:
        router.use(auth_middleware(auth_service, excluded_paths=['/login', '/register']))
    """
    excluded_paths = excluded_paths or []

    def middleware(request: Request, next_handler: Callable) -> Response:
        # Skip auth for excluded paths
        if request.path in excluded_paths:
            return next_handler(request)

        # Check for token
        auth_header = request.authorization
        if not auth_header or not auth_header.startswith('Bearer '):
            return Response.error('Unauthorized', 401)

        token = auth_header[7:]  # Remove 'Bearer ' prefix

        # Validate token
        payload = auth_service.token_manager.validate_token(token)
        if not payload:
            return Response.error('Invalid or expired token', 401)

        # Attach user info to request
        request.headers['X-User-ID'] = payload.get('sub')
        request.headers['X-User-Roles'] = ','.join(payload.get('roles', []))

        return next_handler(request)

    return middleware


def rate_limit_middleware(
    max_requests: int = 100,
    window_seconds: int = 60
) -> Middleware:
    """
    Rate limiting middleware.

    Usage:
        router.use(rate_limit_middleware(max_requests=100, window_seconds=60))
    """
    from collections import defaultdict
    import time

    request_counts: Dict[str, List[float]] = defaultdict(list)

    def middleware(request: Request, next_handler: Callable) -> Response:
        client_ip = request.headers.get('X-Forwarded-For', 'unknown')
        now = time.time()

        # Clean old requests
        request_counts[client_ip] = [
            t for t in request_counts[client_ip]
            if now - t < window_seconds
        ]

        # Check rate limit
        if len(request_counts[client_ip]) >= max_requests:
            return Response.json({
                'error': 'Rate limit exceeded',
                'retry_after': window_seconds
            }, status_code=429)

        # Record request
        request_counts[client_ip].append(now)

        response = next_handler(request)

        # Add rate limit headers
        response.headers['X-RateLimit-Limit'] = str(max_requests)
        response.headers['X-RateLimit-Remaining'] = str(max_requests - len(request_counts[client_ip]))

        return response

    return middleware
