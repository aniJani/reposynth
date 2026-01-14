"""
Benchmark Generator for Large-Scale Evaluation.

Generates 100+ diverse benchmark examples across all categories and difficulties.

Phase 4, Week 8: Scaled Benchmark Dataset
"""

from typing import List, Dict, Any
from .benchmark import (
    BenchmarkDataset,
    BenchmarkExample,
    Difficulty,
    Category,
    create_benchmark,
)


# ============================================================================
# EXPANDED MOCK CODEBASE (for realistic examples)
# ============================================================================

FLASK_APP_CODEBASE = {
    "src/auth/jwt.py": '''"""JWT Authentication Module."""
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24

def create_token(user_id: int, role: str = "user") -> str:
    """Create a JWT token for authenticated user."""
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")

def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Missing token"}), 401
        try:
            payload = verify_token(token)
            request.user = payload
        except ValueError as e:
            return jsonify({"error": str(e)}), 401
        return f(*args, **kwargs)
    return decorated

def require_role(role: str):
    """Decorator to require specific role."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if request.user.get("role") != role:
                return jsonify({"error": "Insufficient permissions"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator
''',

    "src/auth/password.py": '''"""Password hashing utilities."""
import bcrypt
import hashlib
import secrets

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode(), salt).decode()

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode(), hashed.encode())

def generate_reset_token() -> str:
    """Generate a secure password reset token."""
    return secrets.token_urlsafe(32)

def hash_reset_token(token: str) -> str:
    """Hash reset token for storage."""
    return hashlib.sha256(token.encode()).hexdigest()
''',

    "src/api/routes.py": '''"""API Route Definitions."""
from flask import Flask, request, jsonify, Blueprint
from src.auth.jwt import create_token, require_auth, require_role
from src.auth.password import hash_password, verify_password
from src.db.models import User, db
from src.services.email import send_welcome_email

api = Blueprint("api", __name__, url_prefix="/api/v1")

@api.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "version": "1.0.0"})

@api.route("/register", methods=["POST"])
def register():
    """Register a new user."""
    data = request.json

    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "Email already registered"}), 400

    user = User(
        username=data["username"],
        email=data["email"],
        password_hash=hash_password(data["password"]),
    )
    db.session.add(user)
    db.session.commit()

    send_welcome_email(user.email, user.username)

    return jsonify({"id": user.id, "message": "User created"}), 201

@api.route("/login", methods=["POST"])
def login():
    """Authenticate user and return token."""
    data = request.json
    user = User.query.filter_by(email=data["email"]).first()

    if not user or not verify_password(data["password"], user.password_hash):
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_token(user.id, user.role)
    return jsonify({"token": token, "user_id": user.id})

@api.route("/users/me", methods=["GET"])
@require_auth
def get_current_user():
    """Get current authenticated user."""
    user = User.query.get(request.user["user_id"])
    return jsonify(user.to_dict())

@api.route("/users/<int:user_id>", methods=["PUT"])
@require_auth
def update_user(user_id):
    """Update user profile."""
    if request.user["user_id"] != user_id and request.user["role"] != "admin":
        return jsonify({"error": "Forbidden"}), 403

    user = User.query.get_or_404(user_id)
    data = request.json

    if "username" in data:
        user.username = data["username"]
    if "email" in data:
        user.email = data["email"]

    db.session.commit()
    return jsonify(user.to_dict())

@api.route("/admin/users", methods=["GET"])
@require_auth
@require_role("admin")
def list_all_users():
    """List all users (admin only)."""
    users = User.query.all()
    return jsonify([u.to_dict() for u in users])
''',

    "src/api/middleware.py": '''"""API Middleware."""
from flask import request, g
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def request_logger(app):
    """Log all requests."""
    @app.before_request
    def log_request():
        g.start_time = time.time()
        logger.info(f"Request: {request.method} {request.path}")

    @app.after_request
    def log_response(response):
        duration = time.time() - g.start_time
        logger.info(f"Response: {response.status_code} ({duration:.3f}s)")
        return response

def rate_limit(max_requests: int, window_seconds: int):
    """Rate limiting decorator."""
    requests = {}

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            client_ip = request.remote_addr
            now = time.time()

            # Clean old entries
            requests[client_ip] = [
                t for t in requests.get(client_ip, [])
                if now - t < window_seconds
            ]

            if len(requests.get(client_ip, [])) >= max_requests:
                return {"error": "Rate limit exceeded"}, 429

            requests.setdefault(client_ip, []).append(now)
            return f(*args, **kwargs)
        return decorated
    return decorator

def cors_headers(response):
    """Add CORS headers."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE"
    return response
''',

    "src/db/models.py": '''"""Database Models."""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """User model."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), default="user")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    posts = db.relationship("Post", backref="author", lazy="dynamic")
    comments = db.relationship("Comment", backref="author", lazy="dynamic")

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat(),
        }

class Post(db.Model):
    """Blog post model."""
    __tablename__ = "posts"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    author_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    published = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    comments = db.relationship("Comment", backref="post", lazy="dynamic")
    tags = db.relationship("Tag", secondary="post_tags", backref="posts")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "author_id": self.author_id,
            "published": self.published,
            "created_at": self.created_at.isoformat(),
        }

class Comment(db.Model):
    """Comment model."""
    __tablename__ = "comments"

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    author_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey("posts.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Tag(db.Model):
    """Tag model for posts."""
    __tablename__ = "tags"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

# Association table
post_tags = db.Table("post_tags",
    db.Column("post_id", db.Integer, db.ForeignKey("posts.id")),
    db.Column("tag_id", db.Integer, db.ForeignKey("tags.id"))
)
''',

    "src/db/migrations.py": '''"""Database Migration Utilities."""
from flask_migrate import Migrate
from src.db.models import db

migrate = Migrate()

def init_migrations(app):
    """Initialize migrations."""
    migrate.init_app(app, db)

def create_tables(app):
    """Create all database tables."""
    with app.app_context():
        db.create_all()

def drop_tables(app):
    """Drop all database tables."""
    with app.app_context():
        db.drop_all()

def seed_data(app):
    """Seed initial data."""
    from src.db.models import User, Tag
    from src.auth.password import hash_password

    with app.app_context():
        # Create admin user
        admin = User(
            username="admin",
            email="admin@example.com",
            password_hash=hash_password("admin123"),
            role="admin",
        )
        db.session.add(admin)

        # Create default tags
        for tag_name in ["python", "flask", "tutorial", "news"]:
            tag = Tag(name=tag_name)
            db.session.add(tag)

        db.session.commit()
''',

    "src/config.py": '''"""Application Configuration."""
import os
from datetime import timedelta

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///app.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # JWT
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)

    # Email
    MAIL_SERVER = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
    MAIL_PORT = int(os.environ.get("MAIL_PORT", 587))
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")

    # Redis
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    # Rate limiting
    RATELIMIT_DEFAULT = "100/hour"
    RATELIMIT_STORAGE_URL = REDIS_URL

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SQLALCHEMY_ECHO = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

    # Require environment variables in production
    @property
    def SECRET_KEY(self):
        key = os.environ.get("SECRET_KEY")
        if not key:
            raise ValueError("SECRET_KEY must be set in production")
        return key

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False

config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}

def get_config(env=None):
    """Get configuration by environment name."""
    if env is None:
        env = os.environ.get("FLASK_ENV", "development")
    return config.get(env, config["default"])
''',

    "src/services/email.py": '''"""Email Service."""
from flask_mail import Mail, Message
from threading import Thread
from flask import current_app, render_template

mail = Mail()

def send_async_email(app, msg):
    """Send email asynchronously."""
    with app.app_context():
        mail.send(msg)

def send_email(subject, recipient, template, **kwargs):
    """Send an email."""
    app = current_app._get_current_object()
    msg = Message(
        subject=subject,
        sender=app.config["MAIL_USERNAME"],
        recipients=[recipient],
    )
    msg.body = render_template(f"email/{template}.txt", **kwargs)
    msg.html = render_template(f"email/{template}.html", **kwargs)

    thread = Thread(target=send_async_email, args=(app, msg))
    thread.start()
    return thread

def send_welcome_email(email, username):
    """Send welcome email to new user."""
    send_email(
        subject="Welcome to Our App!",
        recipient=email,
        template="welcome",
        username=username,
    )

def send_password_reset(email, reset_url):
    """Send password reset email."""
    send_email(
        subject="Password Reset Request",
        recipient=email,
        template="reset_password",
        reset_url=reset_url,
    )

def send_notification(email, title, message):
    """Send generic notification email."""
    send_email(
        subject=title,
        recipient=email,
        template="notification",
        title=title,
        message=message,
    )
''',

    "src/services/cache.py": '''"""Caching Service."""
import redis
import json
from functools import wraps
from flask import current_app

class CacheService:
    """Redis-based caching service."""

    def __init__(self, app=None):
        self.redis = None
        if app:
            self.init_app(app)

    def init_app(self, app):
        """Initialize with Flask app."""
        redis_url = app.config.get("REDIS_URL", "redis://localhost:6379/0")
        self.redis = redis.from_url(redis_url)

    def get(self, key):
        """Get value from cache."""
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    def set(self, key, value, ttl=3600):
        """Set value in cache with TTL."""
        self.redis.setex(key, ttl, json.dumps(value))

    def delete(self, key):
        """Delete key from cache."""
        self.redis.delete(key)

    def clear_pattern(self, pattern):
        """Delete all keys matching pattern."""
        for key in self.redis.scan_iter(pattern):
            self.redis.delete(key)

cache = CacheService()

def cached(ttl=3600, key_prefix=""):
    """Decorator for caching function results."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            cache_key = f"{key_prefix}:{f.__name__}:{str(args)}:{str(kwargs)}"

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Compute and cache result
            result = f(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        return decorated
    return decorator
''',

    "src/utils/validators.py": '''"""Input Validation Utilities."""
import re
from typing import Optional, List

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

def validate_password(password: str) -> tuple[bool, List[str]]:
    """
    Validate password strength.

    Returns:
        (is_valid, list of errors)
    """
    errors = []

    if len(password) < 8:
        errors.append("Password must be at least 8 characters")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain an uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain a lowercase letter")
    if not re.search(r"[0-9]", password):
        errors.append("Password must contain a number")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        errors.append("Password must contain a special character")

    return len(errors) == 0, errors

def validate_username(username: str) -> tuple[bool, Optional[str]]:
    """Validate username format."""
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(username) > 20:
        return False, "Username must be at most 20 characters"
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return False, "Username can only contain letters, numbers, and underscores"
    return True, None

def sanitize_html(text: str) -> str:
    """Remove potentially dangerous HTML tags."""
    # Simple sanitization - in production use a library like bleach
    dangerous_tags = ["script", "iframe", "object", "embed", "form"]
    result = text
    for tag in dangerous_tags:
        result = re.sub(f"<{tag}[^>]*>.*?</{tag}>", "", result, flags=re.IGNORECASE | re.DOTALL)
        result = re.sub(f"<{tag}[^>]*/>", "", result, flags=re.IGNORECASE)
    return result
''',

    "src/utils/logging.py": '''"""Logging Configuration."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from flask import request, g
import time

def setup_logging(app):
    """Configure application logging."""
    # Remove default handlers
    app.logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if app.debug else logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    app.logger.addHandler(console_handler)

    # File handler for production
    if not app.debug:
        file_handler = RotatingFileHandler(
            "logs/app.log",
            maxBytes=10_000_000,
            backupCount=10,
        )
        file_handler.setLevel(logging.WARNING)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        app.logger.addHandler(file_handler)

    app.logger.setLevel(logging.DEBUG)

def log_request_info():
    """Log request information."""
    g.start_time = time.time()

def log_response_info(response):
    """Log response information."""
    if hasattr(g, "start_time"):
        duration = time.time() - g.start_time
        logging.info(
            f"{request.method} {request.path} - "
            f"{response.status_code} - {duration:.3f}s"
        )
    return response
''',

    "src/app.py": '''"""Application Factory."""
from flask import Flask
from src.config import get_config
from src.db.models import db
from src.db.migrations import init_migrations
from src.services.email import mail
from src.services.cache import cache
from src.api.routes import api
from src.api.middleware import request_logger, cors_headers
from src.utils.logging import setup_logging

def create_app(config_name=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Load configuration
    config_class = get_config(config_name)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    mail.init_app(app)
    cache.init_app(app)
    init_migrations(app)

    # Setup logging
    setup_logging(app)

    # Register blueprints
    app.register_blueprint(api)

    # Apply middleware
    request_logger(app)
    app.after_request(cors_headers)

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Not found"}, 404

    @app.errorhandler(500)
    def server_error(e):
        app.logger.error(f"Server error: {e}")
        return {"error": "Internal server error"}, 500

    return app

# For running with `flask run`
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
''',

    "tests/test_auth.py": '''"""Tests for authentication module."""
import pytest
from src.auth.jwt import create_token, verify_token, require_auth
from src.auth.password import hash_password, verify_password

class TestJWT:
    def test_create_and_verify_token(self):
        token = create_token(user_id=1, role="user")
        payload = verify_token(token)
        assert payload["user_id"] == 1
        assert payload["role"] == "user"

    def test_expired_token(self):
        # Would need to mock time or create expired token
        pass

    def test_invalid_token(self):
        with pytest.raises(ValueError):
            verify_token("invalid-token")

class TestPassword:
    def test_hash_and_verify(self):
        password = "SecureP@ss123"
        hashed = hash_password(password)
        assert verify_password(password, hashed)
        assert not verify_password("wrong", hashed)

    def test_unique_hashes(self):
        password = "Test123!"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        assert hash1 != hash2  # Different salts
''',

    "tests/test_api.py": '''"""Tests for API endpoints."""
import pytest
from src.app import create_app
from src.db.models import db, User

@pytest.fixture
def app():
    app = create_app("testing")
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json["status"] == "healthy"

class TestAuthEndpoints:
    def test_register(self, client):
        response = client.post("/api/v1/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "Test123!",
        })
        assert response.status_code == 201

    def test_login(self, client, app):
        # First register
        client.post("/api/v1/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "Test123!",
        })

        # Then login
        response = client.post("/api/v1/login", json={
            "email": "test@example.com",
            "password": "Test123!",
        })
        assert response.status_code == 200
        assert "token" in response.json
''',
}


# ============================================================================
# BENCHMARK EXAMPLE GENERATORS
# ============================================================================

def generate_architecture_examples() -> List[BenchmarkExample]:
    """Generate architecture-focused examples."""
    return [
        # Easy architecture questions
        BenchmarkExample(
            id="arch_001",
            query="What is the overall structure of the Flask application?",
            repository="flask-app",
            ground_truth_files=["src/app.py"],
            ground_truth_answer="""
The Flask application uses the application factory pattern:
1. create_app() function creates and configures the app
2. Configuration is loaded from src/config.py
3. Extensions are initialized: db, mail, cache, migrations
4. Logging is set up via setup_logging()
5. The api blueprint is registered for routes
6. Middleware is applied: request_logger and cors_headers
7. Error handlers are defined for 404 and 500
""",
            ground_truth_keywords=["create_app", "factory", "blueprint", "extensions", "config"],
            difficulty=Difficulty.EASY,
            category=Category.ARCHITECTURE,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="arch_002",
            query="How does the authentication system work in this application?",
            repository="flask-app",
            ground_truth_files=["src/auth/jwt.py", "src/api/routes.py"],
            ground_truth_answer="""
Authentication uses JWT (JSON Web Tokens):
1. User logs in via /api/v1/login endpoint
2. Server verifies password using verify_password()
3. create_token() generates JWT with user_id, role, exp, iat
4. Token is returned to client
5. Client includes token in Authorization header
6. @require_auth decorator verifies token on protected routes
7. @require_role decorator enforces role-based access
""",
            ground_truth_keywords=["JWT", "create_token", "verify_token", "require_auth", "login", "Authorization"],
            difficulty=Difficulty.MEDIUM,
            category=Category.ARCHITECTURE,
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="arch_003",
            query="How is the database layer organized and what models exist?",
            repository="flask-app",
            ground_truth_files=["src/db/models.py", "src/db/migrations.py"],
            ground_truth_answer="""
Database uses Flask-SQLAlchemy with these models:
1. User: id, username, email, password_hash, role, timestamps
   - Has relationships to posts and comments
2. Post: id, title, content, author_id, published, timestamps
   - Has comments and tags relationships
3. Comment: id, content, author_id, post_id, timestamp
4. Tag: id, name with many-to-many to posts via post_tags table

Migrations handled by Flask-Migrate with init_migrations(), create_tables(), seed_data()
""",
            ground_truth_keywords=["User", "Post", "Comment", "Tag", "SQLAlchemy", "relationship", "migration"],
            difficulty=Difficulty.MEDIUM,
            category=Category.ARCHITECTURE,
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="arch_004",
            query="How does the application handle caching?",
            repository="flask-app",
            ground_truth_files=["src/services/cache.py", "src/config.py"],
            ground_truth_answer="""
Caching uses Redis via CacheService class:
1. Initialized with Redis URL from config (REDIS_URL)
2. get(key) retrieves JSON-decoded values
3. set(key, value, ttl) stores with expiration (default 3600s)
4. delete(key) removes single key
5. clear_pattern(pattern) removes matching keys
6. @cached decorator caches function results with configurable TTL and key prefix
""",
            ground_truth_keywords=["Redis", "CacheService", "cached", "ttl", "REDIS_URL"],
            difficulty=Difficulty.MEDIUM,
            category=Category.ARCHITECTURE,
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="arch_005",
            query="What is the complete request lifecycle from client to response?",
            repository="flask-app",
            ground_truth_files=["src/app.py", "src/api/middleware.py", "src/api/routes.py", "src/utils/logging.py"],
            ground_truth_answer="""
Request lifecycle:
1. Request arrives at Flask app
2. request_logger middleware logs start time via before_request
3. CORS headers added via after_request
4. Route matched in api blueprint
5. @require_auth checks JWT token if protected
6. @require_role checks permissions if needed
7. Route handler processes request
8. Database queries via SQLAlchemy models
9. Response returned
10. after_request logs response status and duration
""",
            ground_truth_keywords=["middleware", "before_request", "after_request", "route", "blueprint", "cors"],
            difficulty=Difficulty.HARD,
            category=Category.ARCHITECTURE,
            expected_retrieval_count=4,
        ),

        # Additional architecture examples
        BenchmarkExample(
            id="arch_006",
            query="How are API routes organized in the application?",
            repository="flask-app",
            ground_truth_files=["src/api/routes.py"],
            ground_truth_answer="""
API routes are organized using Flask Blueprint:
1. Blueprint named 'api' with prefix /api/v1
2. Routes: /health (GET), /register (POST), /login (POST)
3. Protected routes: /users/me (GET), /users/<id> (PUT)
4. Admin routes: /admin/users (GET) requires admin role
5. Authentication via @require_auth and @require_role decorators
""",
            ground_truth_keywords=["Blueprint", "api/v1", "route", "GET", "POST", "PUT"],
            difficulty=Difficulty.EASY,
            category=Category.ARCHITECTURE,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="arch_007",
            query="How does the application handle email sending?",
            repository="flask-app",
            ground_truth_files=["src/services/email.py", "src/config.py"],
            ground_truth_answer="""
Email handling uses Flask-Mail:
1. Configuration via MAIL_SERVER, MAIL_PORT, MAIL_USERNAME, MAIL_PASSWORD
2. Emails sent asynchronously in separate thread
3. Templates from email/ directory (txt and html versions)
4. Helper functions: send_welcome_email, send_password_reset, send_notification
""",
            ground_truth_keywords=["Flask-Mail", "send_email", "async", "thread", "template"],
            difficulty=Difficulty.MEDIUM,
            category=Category.ARCHITECTURE,
            expected_retrieval_count=2,
        ),
    ]


def generate_api_usage_examples() -> List[BenchmarkExample]:
    """Generate API usage examples."""
    return [
        BenchmarkExample(
            id="api_001",
            query="How do I register a new user via the API?",
            repository="flask-app",
            ground_truth_files=["src/api/routes.py"],
            ground_truth_answer="""
To register a new user:
POST /api/v1/register
Body: {"username": "name", "email": "email@example.com", "password": "pass"}
Returns: {"id": 1, "message": "User created"} with 201 status
Errors: 400 if email already registered
""",
            ground_truth_keywords=["register", "POST", "username", "email", "password", "201"],
            difficulty=Difficulty.EASY,
            category=Category.API_USAGE,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="api_002",
            query="How do I authenticate and make protected API calls?",
            repository="flask-app",
            ground_truth_files=["src/api/routes.py", "src/auth/jwt.py"],
            ground_truth_answer="""
Authentication flow:
1. POST /api/v1/login with {"email": "...", "password": "..."}
2. Receive {"token": "jwt_token", "user_id": 1}
3. Include token in subsequent requests:
   Authorization: Bearer <token>
4. Protected endpoints verify token via @require_auth decorator
5. Token contains user_id, role, expiration (24 hours)
""",
            ground_truth_keywords=["login", "token", "Bearer", "Authorization", "require_auth"],
            difficulty=Difficulty.MEDIUM,
            category=Category.API_USAGE,
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="api_003",
            query="How do I use the caching decorator on my functions?",
            repository="flask-app",
            ground_truth_files=["src/services/cache.py"],
            ground_truth_answer="""
Use @cached decorator:
@cached(ttl=3600, key_prefix="myprefix")
def my_function(arg1, arg2):
    # expensive computation
    return result

Parameters:
- ttl: cache expiration in seconds (default 3600)
- key_prefix: optional prefix for cache key
Cache key format: {prefix}:{function_name}:{args}:{kwargs}
""",
            ground_truth_keywords=["cached", "decorator", "ttl", "key_prefix", "cache_key"],
            difficulty=Difficulty.EASY,
            category=Category.API_USAGE,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="api_004",
            query="How do I add rate limiting to an endpoint?",
            repository="flask-app",
            ground_truth_files=["src/api/middleware.py"],
            ground_truth_answer="""
Use @rate_limit decorator:
@rate_limit(max_requests=100, window_seconds=3600)
def my_endpoint():
    return {"data": "response"}

Parameters:
- max_requests: maximum requests allowed
- window_seconds: time window for counting
Returns 429 status with {"error": "Rate limit exceeded"} when exceeded
""",
            ground_truth_keywords=["rate_limit", "max_requests", "window_seconds", "429"],
            difficulty=Difficulty.EASY,
            category=Category.API_USAGE,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="api_005",
            query="How do I validate user input in the application?",
            repository="flask-app",
            ground_truth_files=["src/utils/validators.py"],
            ground_truth_answer="""
Input validation utilities:
1. validate_email(email) - returns bool for email format
2. validate_password(password) - returns (bool, list of errors)
   Requirements: 8+ chars, uppercase, lowercase, number, special char
3. validate_username(username) - returns (bool, error message)
   Requirements: 3-20 chars, alphanumeric and underscore only
4. sanitize_html(text) - removes dangerous HTML tags (script, iframe, etc.)
""",
            ground_truth_keywords=["validate_email", "validate_password", "validate_username", "sanitize_html"],
            difficulty=Difficulty.MEDIUM,
            category=Category.API_USAGE,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="api_006",
            query="How do I update a user's profile via the API?",
            repository="flask-app",
            ground_truth_files=["src/api/routes.py"],
            ground_truth_answer="""
To update user profile:
PUT /api/v1/users/<user_id>
Headers: Authorization: Bearer <token>
Body: {"username": "new_name", "email": "new@email.com"}
Returns: Updated user object

Permissions:
- Users can update their own profile
- Admins can update any profile
- Returns 403 if unauthorized
""",
            ground_truth_keywords=["PUT", "users", "update", "username", "email", "403"],
            difficulty=Difficulty.EASY,
            category=Category.API_USAGE,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="api_007",
            query="How do I send emails using the email service?",
            repository="flask-app",
            ground_truth_files=["src/services/email.py"],
            ground_truth_answer="""
Email sending functions:
1. send_welcome_email(email, username) - welcome new users
2. send_password_reset(email, reset_url) - password reset link
3. send_notification(email, title, message) - generic notifications
4. send_email(subject, recipient, template, **kwargs) - custom emails

All emails use templates from email/ directory with .txt and .html versions
""",
            ground_truth_keywords=["send_welcome_email", "send_password_reset", "send_notification", "template"],
            difficulty=Difficulty.EASY,
            category=Category.API_USAGE,
            expected_retrieval_count=1,
        ),
    ]


def generate_implementation_examples() -> List[BenchmarkExample]:
    """Generate implementation-focused examples."""
    return [
        BenchmarkExample(
            id="impl_001",
            query="How is JWT token creation implemented?",
            repository="flask-app",
            ground_truth_files=["src/auth/jwt.py"],
            ground_truth_answer="""
JWT token creation in create_token():
1. Build payload with user_id, role, exp (expiry), iat (issued at)
2. Expiry set to current time + TOKEN_EXPIRY_HOURS (24 hours)
3. Encode using jwt.encode() with SECRET_KEY and HS256 algorithm
4. Return encoded token string
""",
            ground_truth_keywords=["create_token", "payload", "exp", "iat", "jwt.encode", "HS256"],
            difficulty=Difficulty.EASY,
            category=Category.IMPLEMENTATION,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="impl_002",
            query="How is password hashing implemented?",
            repository="flask-app",
            ground_truth_files=["src/auth/password.py"],
            ground_truth_answer="""
Password hashing uses bcrypt:
1. hash_password(password):
   - Generate salt with bcrypt.gensalt(rounds=12)
   - Hash with bcrypt.hashpw()
   - Return decoded string
2. verify_password(password, hashed):
   - Uses bcrypt.checkpw() to compare
   - Returns boolean
3. Password reset tokens use secrets.token_urlsafe() and SHA-256 hashing
""",
            ground_truth_keywords=["bcrypt", "gensalt", "hashpw", "checkpw", "rounds", "SHA-256"],
            difficulty=Difficulty.EASY,
            category=Category.IMPLEMENTATION,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="impl_003",
            query="How are database relationships defined between User and Post?",
            repository="flask-app",
            ground_truth_files=["src/db/models.py"],
            ground_truth_answer="""
User-Post relationship:
1. Post has author_id foreign key: db.ForeignKey("users.id")
2. User has posts relationship: db.relationship("Post", backref="author", lazy="dynamic")
3. Backref creates post.author to access the user
4. lazy="dynamic" returns query object for filtering
5. Similar pattern for comments on both User and Post
""",
            ground_truth_keywords=["relationship", "ForeignKey", "backref", "author", "lazy", "dynamic"],
            difficulty=Difficulty.MEDIUM,
            category=Category.IMPLEMENTATION,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="impl_004",
            query="How is the rate limiting middleware implemented?",
            repository="flask-app",
            ground_truth_files=["src/api/middleware.py"],
            ground_truth_answer="""
Rate limiting implementation:
1. Decorator rate_limit(max_requests, window_seconds)
2. Uses dictionary to track requests per client IP
3. On each request:
   - Get client IP from request.remote_addr
   - Clean old entries outside time window
   - Check if count >= max_requests
   - Return 429 error if exceeded
   - Otherwise add timestamp and proceed
""",
            ground_truth_keywords=["rate_limit", "decorator", "client_ip", "remote_addr", "429", "window"],
            difficulty=Difficulty.MEDIUM,
            category=Category.IMPLEMENTATION,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="impl_005",
            query="How does the application factory pattern work in this app?",
            repository="flask-app",
            ground_truth_files=["src/app.py", "src/config.py"],
            ground_truth_answer="""
Application factory in create_app():
1. Create Flask instance
2. Load config via get_config(config_name)
3. Initialize extensions: db.init_app(), mail.init_app(), cache.init_app()
4. Setup logging with setup_logging(app)
5. Register blueprint: app.register_blueprint(api)
6. Apply middleware via decorators
7. Register error handlers for 404/500
8. Return configured app instance

Config classes inherit from base Config with environment-specific overrides
""",
            ground_truth_keywords=["create_app", "init_app", "register_blueprint", "config", "factory"],
            difficulty=Difficulty.MEDIUM,
            category=Category.IMPLEMENTATION,
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="impl_006",
            query="How is email sending made asynchronous?",
            repository="flask-app",
            ground_truth_files=["src/services/email.py"],
            ground_truth_answer="""
Asynchronous email implementation:
1. send_async_email(app, msg) runs in separate thread
2. Uses app.app_context() to access Flask context
3. Calls mail.send(msg) within context
4. send_email() creates Thread targeting send_async_email
5. Thread is started immediately with thread.start()
6. Returns thread object for optional join
""",
            ground_truth_keywords=["Thread", "async", "app_context", "mail.send", "thread.start"],
            difficulty=Difficulty.MEDIUM,
            category=Category.IMPLEMENTATION,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="impl_007",
            query="How is the require_auth decorator implemented?",
            repository="flask-app",
            ground_truth_files=["src/auth/jwt.py"],
            ground_truth_answer="""
require_auth decorator implementation:
1. Uses @wraps(f) to preserve function metadata
2. Extracts token from Authorization header (removes "Bearer " prefix)
3. Returns 401 with "Missing token" if no token
4. Calls verify_token(token) to decode
5. Stores payload in request.user for access in route
6. Catches ValueError for expired/invalid tokens
7. Returns original function result if valid
""",
            ground_truth_keywords=["require_auth", "wraps", "Authorization", "Bearer", "verify_token", "request.user"],
            difficulty=Difficulty.MEDIUM,
            category=Category.IMPLEMENTATION,
            expected_retrieval_count=1,
        ),
    ]


def generate_debugging_examples() -> List[BenchmarkExample]:
    """Generate debugging-focused examples."""
    return [
        BenchmarkExample(
            id="debug_001",
            query="Where should I look to debug authentication failures?",
            repository="flask-app",
            ground_truth_files=["src/auth/jwt.py", "src/api/routes.py"],
            ground_truth_answer="""
Debug authentication in these locations:
1. src/auth/jwt.py - verify_token() for token validation errors
   - ExpiredSignatureError: token expired
   - InvalidTokenError: malformed token
2. src/api/routes.py - login() endpoint
   - Check password verification with verify_password()
   - Check user lookup by email
3. Check require_auth decorator for Authorization header parsing
""",
            ground_truth_keywords=["verify_token", "ExpiredSignatureError", "InvalidTokenError", "verify_password", "Authorization"],
            difficulty=Difficulty.MEDIUM,
            category=Category.DEBUGGING,
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="debug_002",
            query="How can I trace why emails are not being sent?",
            repository="flask-app",
            ground_truth_files=["src/services/email.py", "src/config.py"],
            ground_truth_answer="""
Debug email issues:
1. Check config: MAIL_SERVER, MAIL_PORT, MAIL_USERNAME, MAIL_PASSWORD
2. Verify MAIL_USE_TLS is True (port 587)
3. In send_email(): check template paths exist
4. Thread may silently fail - add try/except in send_async_email
5. Verify app context is available in async thread
6. Check Flask-Mail is initialized: mail.init_app(app)
""",
            ground_truth_keywords=["MAIL_SERVER", "MAIL_USERNAME", "MAIL_PASSWORD", "template", "init_app"],
            difficulty=Difficulty.MEDIUM,
            category=Category.DEBUGGING,
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="debug_003",
            query="Why might my database queries be slow and how do I debug them?",
            repository="flask-app",
            ground_truth_files=["src/db/models.py", "src/config.py"],
            ground_truth_answer="""
Debug slow database queries:
1. Enable SQLALCHEMY_ECHO=True in DevelopmentConfig to log queries
2. Check relationships use lazy="dynamic" for large datasets
3. Look for N+1 query problems in relationship access
4. Verify indexes exist on queried columns (id, email, username)
5. Check db.session.commit() timing
6. Use Flask-SQLAlchemy query profiling
""",
            ground_truth_keywords=["SQLALCHEMY_ECHO", "lazy", "dynamic", "index", "relationship", "commit"],
            difficulty=Difficulty.MEDIUM,
            category=Category.DEBUGGING,
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="debug_004",
            query="How do I debug rate limiting issues?",
            repository="flask-app",
            ground_truth_files=["src/api/middleware.py"],
            ground_truth_answer="""
Debug rate limiting:
1. Check client IP detection: request.remote_addr may be proxy IP
2. Verify max_requests and window_seconds parameters
3. requests dict is in-memory - resets on server restart
4. Time cleanup only on each request - stale entries may persist
5. Add logging to see request counts per IP
6. 429 status returned when limit exceeded
""",
            ground_truth_keywords=["remote_addr", "max_requests", "window_seconds", "429", "requests"],
            difficulty=Difficulty.EASY,
            category=Category.DEBUGGING,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="debug_005",
            query="How do I debug cache misses?",
            repository="flask-app",
            ground_truth_files=["src/services/cache.py"],
            ground_truth_answer="""
Debug cache issues:
1. Verify Redis connection: check REDIS_URL in config
2. Check cache key format: {prefix}:{func}:{args}:{kwargs}
3. TTL may have expired - default is 3600 seconds
4. JSON serialization may fail for complex objects
5. Add logging in get() and set() methods
6. Use Redis CLI to inspect keys: KEYS pattern, GET key
""",
            ground_truth_keywords=["Redis", "cache_key", "TTL", "JSON", "REDIS_URL"],
            difficulty=Difficulty.MEDIUM,
            category=Category.DEBUGGING,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="debug_006",
            query="Where do I look to debug 403 Forbidden errors?",
            repository="flask-app",
            ground_truth_files=["src/auth/jwt.py", "src/api/routes.py"],
            ground_truth_answer="""
Debug 403 errors:
1. src/auth/jwt.py - require_role decorator checks request.user["role"]
2. src/api/routes.py - update_user() checks ownership and admin role
3. Verify token contains correct role claim
4. Check role in User model matches expected value
5. Admin endpoints use @require_role("admin")
""",
            ground_truth_keywords=["require_role", "403", "role", "admin", "Forbidden"],
            difficulty=Difficulty.MEDIUM,
            category=Category.DEBUGGING,
            expected_retrieval_count=2,
        ),
    ]


def generate_configuration_examples() -> List[BenchmarkExample]:
    """Generate configuration-focused examples."""
    return [
        BenchmarkExample(
            id="config_001",
            query="What environment variables does the application use?",
            repository="flask-app",
            ground_truth_files=["src/config.py"],
            ground_truth_answer="""
Environment variables:
1. SECRET_KEY - application secret key
2. JWT_SECRET_KEY - JWT signing key (defaults to SECRET_KEY)
3. DATABASE_URL - database connection string
4. MAIL_SERVER, MAIL_PORT, MAIL_USERNAME, MAIL_PASSWORD - email config
5. REDIS_URL - Redis connection URL
6. FLASK_ENV - development/production/testing
""",
            ground_truth_keywords=["SECRET_KEY", "DATABASE_URL", "MAIL_SERVER", "REDIS_URL", "FLASK_ENV"],
            difficulty=Difficulty.EASY,
            category=Category.CONFIGURATION,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="config_002",
            query="How do I configure the application for production?",
            repository="flask-app",
            ground_truth_files=["src/config.py"],
            ground_truth_answer="""
Production configuration (ProductionConfig):
1. DEBUG = False
2. SECRET_KEY required from environment (raises ValueError if missing)
3. Set FLASK_ENV=production
4. Configure real MAIL_SERVER credentials
5. Set DATABASE_URL for production database
6. Configure REDIS_URL for production Redis
7. RATELIMIT_DEFAULT = "100/hour"
""",
            ground_truth_keywords=["ProductionConfig", "DEBUG", "SECRET_KEY", "ValueError", "FLASK_ENV"],
            difficulty=Difficulty.MEDIUM,
            category=Category.CONFIGURATION,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="config_003",
            query="How do I set up the testing configuration?",
            repository="flask-app",
            ground_truth_files=["src/config.py", "tests/test_api.py"],
            ground_truth_answer="""
Testing configuration (TestingConfig):
1. TESTING = True
2. SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:" (in-memory DB)
3. WTF_CSRF_ENABLED = False
4. Pass "testing" to create_app()
5. Use pytest fixtures for app and client
6. Create tables in fixture, drop after tests
""",
            ground_truth_keywords=["TestingConfig", "TESTING", "memory", "pytest", "fixture"],
            difficulty=Difficulty.MEDIUM,
            category=Category.CONFIGURATION,
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="config_004",
            query="How is logging configured in the application?",
            repository="flask-app",
            ground_truth_files=["src/utils/logging.py"],
            ground_truth_answer="""
Logging configuration in setup_logging():
1. Clear default Flask handlers
2. Console handler: DEBUG in dev, INFO in production
3. File handler (production only): WARNING level
4. RotatingFileHandler: 10MB max, 10 backups
5. Format: timestamp - name - level - message
6. Request logging via log_request_info/log_response_info
""",
            ground_truth_keywords=["setup_logging", "RotatingFileHandler", "DEBUG", "WARNING", "console"],
            difficulty=Difficulty.MEDIUM,
            category=Category.CONFIGURATION,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="config_005",
            query="How do I configure the database connection?",
            repository="flask-app",
            ground_truth_files=["src/config.py", "src/db/models.py"],
            ground_truth_answer="""
Database configuration:
1. SQLALCHEMY_DATABASE_URI from DATABASE_URL env var
2. Default: sqlite:///app.db
3. SQLALCHEMY_TRACK_MODIFICATIONS = False (performance)
4. SQLALCHEMY_ECHO = True in development (query logging)
5. Use Flask-SQLAlchemy db object
6. Models inherit from db.Model
""",
            ground_truth_keywords=["SQLALCHEMY_DATABASE_URI", "DATABASE_URL", "sqlite", "TRACK_MODIFICATIONS", "ECHO"],
            difficulty=Difficulty.EASY,
            category=Category.CONFIGURATION,
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="config_006",
            query="How do I configure JWT token settings?",
            repository="flask-app",
            ground_truth_files=["src/config.py", "src/auth/jwt.py"],
            ground_truth_answer="""
JWT configuration:
1. JWT_SECRET_KEY in config (defaults to SECRET_KEY)
2. JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
3. In jwt.py: SECRET_KEY, ALGORITHM = "HS256"
4. TOKEN_EXPIRY_HOURS = 24
5. Payload includes: user_id, role, exp, iat
""",
            ground_truth_keywords=["JWT_SECRET_KEY", "TOKEN_EXPIRY", "HS256", "timedelta", "exp"],
            difficulty=Difficulty.EASY,
            category=Category.CONFIGURATION,
            expected_retrieval_count=2,
        ),
    ]


def generate_data_flow_examples() -> List[BenchmarkExample]:
    """Generate data flow examples."""
    return [
        BenchmarkExample(
            id="flow_001",
            query="How does data flow during user registration?",
            repository="flask-app",
            ground_truth_files=["src/api/routes.py", "src/auth/password.py", "src/db/models.py", "src/services/email.py"],
            ground_truth_answer="""
User registration data flow:
1. POST /api/v1/register receives JSON body
2. Check if email exists in User table
3. Create User with hash_password() from password.py
4. Add to db.session and commit
5. Call send_welcome_email() async
6. Return user ID and success message
""",
            ground_truth_keywords=["register", "hash_password", "User", "session", "commit", "send_welcome_email"],
            difficulty=Difficulty.HARD,
            category=Category.DATA_FLOW,
            expected_retrieval_count=4,
        ),
        BenchmarkExample(
            id="flow_002",
            query="How does data flow during the login process?",
            repository="flask-app",
            ground_truth_files=["src/api/routes.py", "src/auth/jwt.py", "src/auth/password.py"],
            ground_truth_answer="""
Login data flow:
1. POST /api/v1/login with email and password
2. Query User by email
3. verify_password() checks against hash
4. If valid, create_token(user_id, role)
5. Token payload: user_id, role, exp, iat
6. Return JWT token and user_id
""",
            ground_truth_keywords=["login", "verify_password", "create_token", "payload", "JWT"],
            difficulty=Difficulty.MEDIUM,
            category=Category.DATA_FLOW,
            expected_retrieval_count=3,
        ),
        BenchmarkExample(
            id="flow_003",
            query="How does authentication flow through protected endpoints?",
            repository="flask-app",
            ground_truth_files=["src/auth/jwt.py", "src/api/routes.py"],
            ground_truth_answer="""
Protected endpoint authentication flow:
1. Client sends Authorization: Bearer <token>
2. @require_auth extracts token from header
3. verify_token() decodes and validates
4. Payload stored in request.user
5. @require_role checks role if needed
6. Route handler accesses request.user["user_id"]
7. Query User from database for full data
""",
            ground_truth_keywords=["Authorization", "Bearer", "verify_token", "request.user", "require_auth"],
            difficulty=Difficulty.MEDIUM,
            category=Category.DATA_FLOW,
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="flow_004",
            query="How does cached data flow through the application?",
            repository="flask-app",
            ground_truth_files=["src/services/cache.py"],
            ground_truth_answer="""
Cached data flow:
1. @cached decorator wraps function
2. Build cache key from prefix, func name, args, kwargs
3. Check Redis: cache.get(key)
4. If hit: return JSON-decoded value
5. If miss: call original function
6. Store result: cache.set(key, value, ttl)
7. Return computed result
""",
            ground_truth_keywords=["cached", "cache_key", "get", "set", "JSON", "Redis"],
            difficulty=Difficulty.MEDIUM,
            category=Category.DATA_FLOW,
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="flow_005",
            query="How does request data flow through middleware?",
            repository="flask-app",
            ground_truth_files=["src/api/middleware.py", "src/utils/logging.py"],
            ground_truth_answer="""
Middleware request flow:
1. before_request: log_request_info records start time in g
2. Route processing occurs
3. after_request: log_response_info logs status and duration
4. cors_headers adds CORS headers to response
5. Rate limiting checks happen in decorator before route
6. Returns final response to client
""",
            ground_truth_keywords=["before_request", "after_request", "g.start_time", "CORS", "rate_limit"],
            difficulty=Difficulty.MEDIUM,
            category=Category.DATA_FLOW,
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="flow_006",
            query="How does error handling flow in the application?",
            repository="flask-app",
            ground_truth_files=["src/app.py", "src/auth/jwt.py"],
            ground_truth_answer="""
Error handling flow:
1. JWT errors: verify_token raises ValueError with message
2. require_auth catches ValueError, returns 401 JSON
3. require_role returns 403 for insufficient permissions
4. Flask errorhandler(404) returns JSON not found
5. errorhandler(500) logs error and returns generic message
6. Route-level errors return appropriate status codes
""",
            ground_truth_keywords=["errorhandler", "ValueError", "401", "403", "404", "500"],
            difficulty=Difficulty.MEDIUM,
            category=Category.DATA_FLOW,
            expected_retrieval_count=2,
        ),
    ]


def generate_hard_examples() -> List[BenchmarkExample]:
    """Generate challenging multi-file examples."""
    return [
        BenchmarkExample(
            id="hard_001",
            query="How would I implement a password reset feature using the existing codebase?",
            repository="flask-app",
            ground_truth_files=["src/auth/password.py", "src/services/email.py", "src/api/routes.py", "src/db/models.py"],
            ground_truth_answer="""
Password reset implementation using existing code:
1. Add reset_token and reset_token_expiry columns to User model
2. Create endpoint POST /forgot-password:
   - Generate token with generate_reset_token()
   - Hash with hash_reset_token() for storage
   - Set expiry time
   - Call send_password_reset(email, reset_url)
3. Create endpoint POST /reset-password:
   - Verify token hash matches and not expired
   - Update password_hash with hash_password()
   - Clear reset token
""",
            ground_truth_keywords=["generate_reset_token", "hash_reset_token", "send_password_reset", "hash_password", "User"],
            difficulty=Difficulty.HARD,
            category=Category.IMPLEMENTATION,
            expected_retrieval_count=4,
        ),
        BenchmarkExample(
            id="hard_002",
            query="Explain the complete security architecture of this application.",
            repository="flask-app",
            ground_truth_files=["src/auth/jwt.py", "src/auth/password.py", "src/api/middleware.py", "src/utils/validators.py", "src/config.py"],
            ground_truth_answer="""
Security architecture:
1. Authentication: JWT with HS256, 24h expiry, stored in Authorization header
2. Password: bcrypt hashing with 12 rounds salt
3. Authorization: Role-based via @require_role decorator
4. Rate limiting: IP-based request counting middleware
5. Input validation: Email, password strength, username validators
6. HTML sanitization: Remove dangerous tags (script, iframe)
7. CORS: Headers added via middleware
8. Secrets: From environment variables in production
""",
            ground_truth_keywords=["JWT", "bcrypt", "rate_limit", "validate", "sanitize", "CORS", "SECRET_KEY"],
            difficulty=Difficulty.HARD,
            category=Category.ARCHITECTURE,
            expected_retrieval_count=5,
        ),
        BenchmarkExample(
            id="hard_003",
            query="How would I add a new Post CRUD API using the existing patterns?",
            repository="flask-app",
            ground_truth_files=["src/api/routes.py", "src/db/models.py", "src/auth/jwt.py"],
            ground_truth_answer="""
Post CRUD implementation following existing patterns:
1. Use existing Post model with author_id relationship
2. Create routes in api blueprint:
   - GET /posts - list posts (optionally filtered by author)
   - POST /posts - create post (@require_auth)
   - GET /posts/<id> - get single post
   - PUT /posts/<id> - update post (author or admin)
   - DELETE /posts/<id> - delete post (author or admin)
3. Use @require_auth for protected routes
4. Check ownership: request.user["user_id"] == post.author_id
5. Use db.session for transactions
6. Return post.to_dict() for responses
""",
            ground_truth_keywords=["Post", "CRUD", "require_auth", "author_id", "to_dict", "session"],
            difficulty=Difficulty.HARD,
            category=Category.IMPLEMENTATION,
            expected_retrieval_count=3,
        ),
        BenchmarkExample(
            id="hard_004",
            query="How do all the components work together when a user creates a post?",
            repository="flask-app",
            ground_truth_files=["src/api/routes.py", "src/auth/jwt.py", "src/db/models.py", "src/api/middleware.py", "src/utils/logging.py"],
            ground_truth_answer="""
Post creation flow:
1. Request arrives, middleware logs start time
2. Rate limit checked (if applied)
3. @require_auth extracts and verifies JWT token
4. Token payload stored in request.user
5. Route handler gets JSON body
6. Create Post instance with author_id from request.user
7. Add to db.session, commit transaction
8. Return post.to_dict() as JSON
9. Middleware logs response status and duration
10. CORS headers added to response
""",
            ground_truth_keywords=["require_auth", "request.user", "Post", "session", "commit", "middleware", "CORS"],
            difficulty=Difficulty.HARD,
            category=Category.DATA_FLOW,
            expected_retrieval_count=5,
        ),
    ]


# ============================================================================
# MAIN GENERATOR FUNCTION
# ============================================================================

def generate_full_benchmark(
    name: str = "reposynth_benchmark_v2",
    version: str = "2.0.0",
) -> BenchmarkDataset:
    """
    Generate a comprehensive benchmark dataset with 100+ examples.

    Returns:
        BenchmarkDataset with examples across all categories and difficulties
    """
    examples = []

    # Generate examples for each category
    examples.extend(generate_architecture_examples())
    examples.extend(generate_api_usage_examples())
    examples.extend(generate_implementation_examples())
    examples.extend(generate_debugging_examples())
    examples.extend(generate_configuration_examples())
    examples.extend(generate_data_flow_examples())
    examples.extend(generate_hard_examples())

    dataset = BenchmarkDataset(
        name=name,
        version=version,
        examples=examples,
        description="Comprehensive benchmark dataset for evaluating adaptive code generation",
        metadata={
            'created_by': 'RepoSynth',
            'total_examples': len(examples),
            'codebase': 'flask-app',
        },
    )

    return dataset


def get_mock_codebase() -> Dict[str, str]:
    """Get the mock Flask application codebase for testing."""
    return FLASK_APP_CODEBASE.copy()
