"""
Database Service Module

Provides database connection pooling, query building, and ORM-like functionality
for interacting with SQL databases.
"""

import sqlite3
from typing import Optional, Dict, List, Any, Tuple, Union, TypeVar, Generic
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime
import threading
import queue
import json


T = TypeVar('T')


@dataclass
class QueryResult:
    """Represents the result of a database query."""
    rows: List[Dict[str, Any]]
    rowcount: int
    lastrowid: Optional[int] = None
    execution_time_ms: float = 0.0
    query: str = ""


@dataclass
class ConnectionConfig:
    """Database connection configuration."""
    database: str
    pool_size: int = 5
    max_overflow: int = 10
    timeout: float = 30.0
    check_same_thread: bool = False


class ConnectionPool:
    """
    Thread-safe database connection pool.

    Manages a pool of reusable database connections to avoid
    the overhead of creating new connections for each query.
    """

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._pool: queue.Queue = queue.Queue(maxsize=config.pool_size)
        self._size = 0
        self._lock = threading.Lock()
        self._local = threading.local()

        # Pre-create connections
        for _ in range(config.pool_size):
            self._add_connection()

    def _add_connection(self) -> None:
        """Create and add a new connection to the pool."""
        conn = sqlite3.connect(
            self.config.database,
            check_same_thread=self.config.check_same_thread,
            timeout=self.config.timeout
        )
        conn.row_factory = sqlite3.Row
        self._pool.put(conn)
        self._size += 1

    def get_connection(self) -> sqlite3.Connection:
        """
        Get a connection from the pool.

        Returns:
            A database connection

        Raises:
            TimeoutError: If no connection available within timeout
        """
        try:
            conn = self._pool.get(timeout=self.config.timeout)
            return conn
        except queue.Empty:
            with self._lock:
                if self._size < self.config.pool_size + self.config.max_overflow:
                    self._add_connection()
                    return self._pool.get(timeout=self.config.timeout)
            raise TimeoutError("Connection pool exhausted")

    def return_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            conn.close()
            with self._lock:
                self._size -= 1

    @contextmanager
    def connection(self):
        """Context manager for getting and returning connections."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break
        self._size = 0


class QueryBuilder:
    """
    Fluent interface for building SQL queries.

    Usage:
        query = (QueryBuilder('users')
            .select('id', 'name', 'email')
            .where('status', '=', 'active')
            .where('created_at', '>', '2024-01-01')
            .order_by('name')
            .limit(10)
            .build())
    """

    def __init__(self, table: str):
        self.table = table
        self._select_cols: List[str] = ['*']
        self._where_clauses: List[Tuple[str, str, Any]] = []
        self._order_by: List[Tuple[str, str]] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._joins: List[str] = []
        self._group_by: List[str] = []
        self._having: List[Tuple[str, str, Any]] = []

    def select(self, *columns: str) -> 'QueryBuilder':
        """Specify columns to select."""
        self._select_cols = list(columns) if columns else ['*']
        return self

    def where(self, column: str, operator: str, value: Any) -> 'QueryBuilder':
        """Add a WHERE clause."""
        self._where_clauses.append((column, operator, value))
        return self

    def where_in(self, column: str, values: List[Any]) -> 'QueryBuilder':
        """Add a WHERE IN clause."""
        self._where_clauses.append((column, 'IN', values))
        return self

    def where_null(self, column: str) -> 'QueryBuilder':
        """Add a WHERE IS NULL clause."""
        self._where_clauses.append((column, 'IS', None))
        return self

    def where_not_null(self, column: str) -> 'QueryBuilder':
        """Add a WHERE IS NOT NULL clause."""
        self._where_clauses.append((column, 'IS NOT', None))
        return self

    def join(self, table: str, on: str, join_type: str = 'INNER') -> 'QueryBuilder':
        """Add a JOIN clause."""
        self._joins.append(f"{join_type} JOIN {table} ON {on}")
        return self

    def left_join(self, table: str, on: str) -> 'QueryBuilder':
        """Add a LEFT JOIN clause."""
        return self.join(table, on, 'LEFT')

    def order_by(self, column: str, direction: str = 'ASC') -> 'QueryBuilder':
        """Add an ORDER BY clause."""
        self._order_by.append((column, direction.upper()))
        return self

    def group_by(self, *columns: str) -> 'QueryBuilder':
        """Add a GROUP BY clause."""
        self._group_by.extend(columns)
        return self

    def having(self, column: str, operator: str, value: Any) -> 'QueryBuilder':
        """Add a HAVING clause."""
        self._having.append((column, operator, value))
        return self

    def limit(self, count: int) -> 'QueryBuilder':
        """Set the LIMIT."""
        self._limit = count
        return self

    def offset(self, count: int) -> 'QueryBuilder':
        """Set the OFFSET."""
        self._offset = count
        return self

    def build(self) -> Tuple[str, List[Any]]:
        """
        Build the SQL query and parameters.

        Returns:
            Tuple of (query_string, parameters)
        """
        params = []

        # SELECT
        cols = ', '.join(self._select_cols)
        query = f"SELECT {cols} FROM {self.table}"

        # JOINs
        if self._joins:
            query += ' ' + ' '.join(self._joins)

        # WHERE
        if self._where_clauses:
            conditions = []
            for col, op, val in self._where_clauses:
                if val is None:
                    conditions.append(f"{col} {op} NULL")
                elif op == 'IN':
                    placeholders = ', '.join(['?' for _ in val])
                    conditions.append(f"{col} IN ({placeholders})")
                    params.extend(val)
                else:
                    conditions.append(f"{col} {op} ?")
                    params.append(val)
            query += ' WHERE ' + ' AND '.join(conditions)

        # GROUP BY
        if self._group_by:
            query += ' GROUP BY ' + ', '.join(self._group_by)

        # HAVING
        if self._having:
            conditions = []
            for col, op, val in self._having:
                conditions.append(f"{col} {op} ?")
                params.append(val)
            query += ' HAVING ' + ' AND '.join(conditions)

        # ORDER BY
        if self._order_by:
            orders = [f"{col} {direction}" for col, direction in self._order_by]
            query += ' ORDER BY ' + ', '.join(orders)

        # LIMIT/OFFSET
        if self._limit is not None:
            query += f' LIMIT {self._limit}'
        if self._offset is not None:
            query += f' OFFSET {self._offset}'

        return query, params


@dataclass
class Model:
    """
    Base class for database models.

    Provides ORM-like functionality for database operations.
    """
    __tablename__: str = ""
    __primary_key__: str = "id"

    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Repository(Generic[T]):
    """
    Generic repository for database operations on models.

    Usage:
        class UserRepository(Repository[User]):
            def __init__(self, db: DatabaseService):
                super().__init__(db, 'users', User)

        repo = UserRepository(db)
        user = repo.find_by_id(1)
        users = repo.find_all(limit=10)
    """

    def __init__(self, db: 'DatabaseService', table: str, model_class: type):
        self.db = db
        self.table = table
        self.model_class = model_class

    def find_by_id(self, id: int) -> Optional[T]:
        """Find a record by its primary key."""
        result = self.db.query_one(
            f"SELECT * FROM {self.table} WHERE id = ?",
            (id,)
        )
        return self._to_model(result) if result else None

    def find_all(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = None
    ) -> List[T]:
        """Find all records with pagination."""
        query = f"SELECT * FROM {self.table}"
        if order_by:
            query += f" ORDER BY {order_by}"
        query += f" LIMIT {limit} OFFSET {offset}"

        results = self.db.query(query)
        return [self._to_model(r) for r in results.rows]

    def find_by(self, **conditions) -> List[T]:
        """Find records matching conditions."""
        builder = QueryBuilder(self.table)
        for col, val in conditions.items():
            builder.where(col, '=', val)

        query, params = builder.build()
        results = self.db.query(query, tuple(params))
        return [self._to_model(r) for r in results.rows]

    def find_one_by(self, **conditions) -> Optional[T]:
        """Find a single record matching conditions."""
        results = self.find_by(**conditions)
        return results[0] if results else None

    def insert(self, model: T) -> T:
        """Insert a new record."""
        data = self._to_dict(model)
        data.pop('id', None)
        data['created_at'] = datetime.utcnow().isoformat()
        data['updated_at'] = data['created_at']

        cols = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {self.table} ({cols}) VALUES ({placeholders})"

        result = self.db.execute(query, tuple(data.values()))
        model.id = result.lastrowid
        return model

    def update(self, model: T) -> T:
        """Update an existing record."""
        if not model.id:
            raise ValueError("Cannot update model without id")

        data = self._to_dict(model)
        data.pop('id')
        data['updated_at'] = datetime.utcnow().isoformat()

        sets = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {self.table} SET {sets} WHERE id = ?"

        params = list(data.values()) + [model.id]
        self.db.execute(query, tuple(params))
        return model

    def delete(self, id: int) -> bool:
        """Delete a record by id."""
        result = self.db.execute(
            f"DELETE FROM {self.table} WHERE id = ?",
            (id,)
        )
        return result.rowcount > 0

    def count(self, **conditions) -> int:
        """Count records matching conditions."""
        builder = QueryBuilder(self.table).select('COUNT(*) as count')
        for col, val in conditions.items():
            builder.where(col, '=', val)

        query, params = builder.build()
        result = self.db.query_one(query, tuple(params))
        return result['count'] if result else 0

    def exists(self, **conditions) -> bool:
        """Check if any records match conditions."""
        return self.count(**conditions) > 0

    def _to_model(self, row: Dict) -> T:
        """Convert a database row to a model instance."""
        if hasattr(self.model_class, '__dataclass_fields__'):
            fields = self.model_class.__dataclass_fields__.keys()
            data = {k: v for k, v in row.items() if k in fields}
            return self.model_class(**data)
        return row

    def _to_dict(self, model: T) -> Dict:
        """Convert a model to a dictionary."""
        if hasattr(model, '__dataclass_fields__'):
            return {k: getattr(model, k) for k in model.__dataclass_fields__.keys()}
        return dict(model)


class DatabaseService:
    """
    Main database service providing high-level database operations.

    Usage:
        db = DatabaseService(ConnectionConfig(database='app.db'))

        # Simple query
        result = db.query("SELECT * FROM users WHERE status = ?", ('active',))

        # Query builder
        builder = db.table('users').select('id', 'name').where('active', '=', True)
        result = db.execute_builder(builder)

        # Transaction
        with db.transaction():
            db.execute("INSERT INTO users ...")
            db.execute("INSERT INTO profiles ...")
    """

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.pool = ConnectionPool(config)
        self._transaction_conn: threading.local = threading.local()

    def query(self, sql: str, params: tuple = None) -> QueryResult:
        """
        Execute a SELECT query and return results.

        Args:
            sql: SQL query string
            params: Query parameters

        Returns:
            QueryResult with rows and metadata
        """
        start = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params or ())

            rows = [dict(row) for row in cursor.fetchall()]

            return QueryResult(
                rows=rows,
                rowcount=len(rows),
                execution_time_ms=(datetime.now() - start).total_seconds() * 1000,
                query=sql
            )

    def query_one(self, sql: str, params: tuple = None) -> Optional[Dict]:
        """Execute a query and return the first row."""
        result = self.query(sql, params)
        return result.rows[0] if result.rows else None

    def execute(self, sql: str, params: tuple = None) -> QueryResult:
        """
        Execute an INSERT/UPDATE/DELETE query.

        Args:
            sql: SQL statement
            params: Query parameters

        Returns:
            QueryResult with rowcount and lastrowid
        """
        start = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params or ())

            if not hasattr(self._transaction_conn, 'conn'):
                conn.commit()

            return QueryResult(
                rows=[],
                rowcount=cursor.rowcount,
                lastrowid=cursor.lastrowid,
                execution_time_ms=(datetime.now() - start).total_seconds() * 1000,
                query=sql
            )

    def execute_many(self, sql: str, params_list: List[tuple]) -> QueryResult:
        """Execute a query multiple times with different parameters."""
        start = datetime.now()
        total_rows = 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, params_list)
            total_rows = cursor.rowcount

            if not hasattr(self._transaction_conn, 'conn'):
                conn.commit()

            return QueryResult(
                rows=[],
                rowcount=total_rows,
                execution_time_ms=(datetime.now() - start).total_seconds() * 1000,
                query=sql
            )

    def table(self, name: str) -> QueryBuilder:
        """Create a query builder for a table."""
        return QueryBuilder(name)

    def execute_builder(self, builder: QueryBuilder) -> QueryResult:
        """Execute a query built with QueryBuilder."""
        query, params = builder.build()
        return self.query(query, tuple(params))

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.

        All queries within the context will be part of the same transaction.
        The transaction is committed on success, rolled back on exception.
        """
        conn = self.pool.get_connection()
        self._transaction_conn.conn = conn

        try:
            yield
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            del self._transaction_conn.conn
            self.pool.return_connection(conn)

    @contextmanager
    def _get_connection(self):
        """Get connection, using transaction connection if in transaction."""
        if hasattr(self._transaction_conn, 'conn'):
            yield self._transaction_conn.conn
        else:
            with self.pool.connection() as conn:
                yield conn

    def close(self) -> None:
        """Close all connections."""
        self.pool.close_all()
