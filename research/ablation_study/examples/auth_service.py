"""
Authentication Service Module

This module provides authentication and authorization functionality
including JWT token management, session handling, and user validation.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import json


@dataclass
class User:
    """Represents an authenticated user."""
    id: str
    username: str
    email: str
    roles: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Represents an active user session."""
    session_id: str
    user_id: str
    token: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_valid: bool = True


class TokenManager:
    """
    Manages JWT-like token generation and validation.

    Tokens are structured as: base64(header).base64(payload).signature
    """

    def __init__(self, secret_key: str, token_expiry_hours: int = 24):
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self._token_cache: Dict[str, Dict] = {}

    def generate_token(self, user: User, additional_claims: Dict = None) -> str:
        """
        Generate a new authentication token for a user.

        Args:
            user: The user to generate token for
            additional_claims: Optional additional JWT claims

        Returns:
            Encoded token string
        """
        now = datetime.utcnow()
        expiry = now + timedelta(hours=self.token_expiry_hours)

        payload = {
            'sub': user.id,
            'username': user.username,
            'email': user.email,
            'roles': user.roles,
            'iat': now.isoformat(),
            'exp': expiry.isoformat(),
            'jti': secrets.token_hex(16),
        }

        if additional_claims:
            payload.update(additional_claims)

        # Create signature
        payload_str = json.dumps(payload, sort_keys=True)
        signature = self._create_signature(payload_str)

        # Encode token
        import base64
        encoded_payload = base64.urlsafe_b64encode(payload_str.encode()).decode()
        token = f"{encoded_payload}.{signature}"

        # Cache token
        self._token_cache[token] = payload

        return token

    def validate_token(self, token: str) -> Optional[Dict]:
        """
        Validate a token and return its payload if valid.

        Args:
            token: The token to validate

        Returns:
            Token payload if valid, None otherwise
        """
        try:
            # Check cache first
            if token in self._token_cache:
                payload = self._token_cache[token]
                if datetime.fromisoformat(payload['exp']) > datetime.utcnow():
                    return payload
                else:
                    del self._token_cache[token]
                    return None

            # Parse token
            parts = token.split('.')
            if len(parts) != 2:
                return None

            encoded_payload, signature = parts

            # Decode and verify
            import base64
            payload_str = base64.urlsafe_b64decode(encoded_payload).decode()

            if self._create_signature(payload_str) != signature:
                return None

            payload = json.loads(payload_str)

            # Check expiry
            if datetime.fromisoformat(payload['exp']) <= datetime.utcnow():
                return None

            return payload

        except Exception:
            return None

    def refresh_token(self, old_token: str) -> Optional[str]:
        """
        Refresh an existing token, generating a new one with extended expiry.

        Args:
            old_token: The token to refresh

        Returns:
            New token if old token is valid, None otherwise
        """
        payload = self.validate_token(old_token)
        if not payload:
            return None

        # Invalidate old token
        if old_token in self._token_cache:
            del self._token_cache[old_token]

        # Create new user from payload
        user = User(
            id=payload['sub'],
            username=payload['username'],
            email=payload['email'],
            roles=payload['roles'],
        )

        return self.generate_token(user)

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token, making it invalid for future use.

        Args:
            token: The token to revoke

        Returns:
            True if token was revoked, False if not found
        """
        if token in self._token_cache:
            del self._token_cache[token]
            return True
        return False

    def _create_signature(self, data: str) -> str:
        """Create HMAC signature for data."""
        return hashlib.sha256(
            f"{data}{self.secret_key}".encode()
        ).hexdigest()


class SessionManager:
    """
    Manages user sessions with support for multiple concurrent sessions.
    """

    def __init__(self, max_sessions_per_user: int = 5, session_timeout_hours: int = 72):
        self.max_sessions_per_user = max_sessions_per_user
        self.session_timeout_hours = session_timeout_hours
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]

    def create_session(
        self,
        user: User,
        token: str,
        ip_address: str = None,
        user_agent: str = None
    ) -> Session:
        """
        Create a new session for a user.

        Args:
            user: The user to create session for
            token: Authentication token for the session
            ip_address: Optional client IP address
            user_agent: Optional client user agent

        Returns:
            Created Session object
        """
        # Check session limit
        if user.id in self._user_sessions:
            existing_sessions = self._user_sessions[user.id]
            if len(existing_sessions) >= self.max_sessions_per_user:
                # Remove oldest session
                oldest_id = existing_sessions[0]
                self.invalidate_session(oldest_id)

        # Create new session
        session = Session(
            session_id=secrets.token_hex(32),
            user_id=user.id,
            token=token,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=self.session_timeout_hours),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Store session
        self._sessions[session.session_id] = session

        if user.id not in self._user_sessions:
            self._user_sessions[user.id] = []
        self._user_sessions[user.id].append(session.session_id)

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by its ID.

        Args:
            session_id: The session ID to look up

        Returns:
            Session if found and valid, None otherwise
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        if not session.is_valid or session.expires_at <= datetime.utcnow():
            self.invalidate_session(session_id)
            return None

        return session

    def get_user_sessions(self, user_id: str) -> List[Session]:
        """
        Get all active sessions for a user.

        Args:
            user_id: The user ID to get sessions for

        Returns:
            List of active sessions
        """
        session_ids = self._user_sessions.get(user_id, [])
        sessions = []

        for sid in session_ids:
            session = self.get_session(sid)
            if session:
                sessions.append(session)

        return sessions

    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a specific session.

        Args:
            session_id: The session ID to invalidate

        Returns:
            True if session was invalidated, False if not found
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.is_valid = False

        # Remove from user's session list
        if session.user_id in self._user_sessions:
            self._user_sessions[session.user_id] = [
                sid for sid in self._user_sessions[session.user_id]
                if sid != session_id
            ]

        del self._sessions[session_id]
        return True

    def invalidate_user_sessions(self, user_id: str) -> int:
        """
        Invalidate all sessions for a user (logout from all devices).

        Args:
            user_id: The user ID to logout

        Returns:
            Number of sessions invalidated
        """
        session_ids = self._user_sessions.get(user_id, []).copy()
        count = 0

        for session_id in session_ids:
            if self.invalidate_session(session_id):
                count += 1

        return count


class AuthService:
    """
    Main authentication service combining token and session management.

    Usage:
        auth = AuthService(secret_key="your-secret-key")

        # Login user
        user = User(id="123", username="john", email="john@example.com")
        result = auth.login(user, ip_address="192.168.1.1")

        # Validate request
        session = auth.validate_request(result['token'], result['session_id'])

        # Logout
        auth.logout(result['session_id'])
    """

    def __init__(
        self,
        secret_key: str,
        token_expiry_hours: int = 24,
        session_timeout_hours: int = 72,
        max_sessions_per_user: int = 5
    ):
        self.token_manager = TokenManager(secret_key, token_expiry_hours)
        self.session_manager = SessionManager(max_sessions_per_user, session_timeout_hours)
        self._user_store: Dict[str, User] = {}

    def register_user(self, user: User) -> None:
        """Register a user in the auth service."""
        self._user_store[user.id] = user

    def login(
        self,
        user: User,
        ip_address: str = None,
        user_agent: str = None,
        additional_claims: Dict = None
    ) -> Dict[str, Any]:
        """
        Perform user login, creating token and session.

        Args:
            user: User to login
            ip_address: Client IP address
            user_agent: Client user agent
            additional_claims: Additional JWT claims

        Returns:
            Dict with token, session_id, and expiry information
        """
        # Update last login
        user.last_login = datetime.utcnow()

        # Generate token
        token = self.token_manager.generate_token(user, additional_claims)

        # Create session
        session = self.session_manager.create_session(
            user, token, ip_address, user_agent
        )

        return {
            'token': token,
            'session_id': session.session_id,
            'expires_at': session.expires_at.isoformat(),
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'roles': user.roles,
            }
        }

    def validate_request(self, token: str, session_id: str = None) -> Optional[Dict]:
        """
        Validate an incoming request's authentication.

        Args:
            token: Authentication token
            session_id: Optional session ID for additional validation

        Returns:
            User info if valid, None otherwise
        """
        # Validate token
        payload = self.token_manager.validate_token(token)
        if not payload:
            return None

        # If session_id provided, validate it too
        if session_id:
            session = self.session_manager.get_session(session_id)
            if not session or session.token != token:
                return None

        return payload

    def logout(self, session_id: str) -> bool:
        """
        Logout a specific session.

        Args:
            session_id: Session to logout

        Returns:
            True if logout successful
        """
        session = self.session_manager.get_session(session_id)
        if session:
            self.token_manager.revoke_token(session.token)
        return self.session_manager.invalidate_session(session_id)

    def logout_all(self, user_id: str) -> int:
        """
        Logout user from all devices.

        Args:
            user_id: User to logout

        Returns:
            Number of sessions logged out
        """
        sessions = self.session_manager.get_user_sessions(user_id)
        for session in sessions:
            self.token_manager.revoke_token(session.token)
        return self.session_manager.invalidate_user_sessions(user_id)

    def check_permission(self, token: str, required_role: str) -> bool:
        """
        Check if the token holder has a specific role.

        Args:
            token: Authentication token
            required_role: Role to check for

        Returns:
            True if user has the role
        """
        payload = self.token_manager.validate_token(token)
        if not payload:
            return False
        return required_role in payload.get('roles', [])
