from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session_factory() -> sessionmaker:
    """For code that needs to open its OWN session outside the normal
    per-request lifecycle — background tasks, mainly, which run after the
    request's `get_db` session has already closed.

    Deliberately a FastAPI dependency (not a bare import of SessionLocal),
    so tests can override it to the same isolated test database `get_db` is
    overridden to — a background task that imported SessionLocal directly
    would silently operate on the real configured database during tests,
    since dependency_overrides only intercepts Depends(get_db)."""
    return SessionLocal


def ensure_schema() -> None:
    """Minimal forward-only migration: `create_all` creates missing tables but
    never adds new columns to existing ones. Diff the models against the live
    schema and ADD COLUMN for anything missing, so a schema change doesn't
    force wiping the database. (Real migrations — Alembic — when we move to
    Postgres.)"""
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                ddl = (
                    f"ALTER TABLE {table.name} ADD COLUMN "
                    f"{column.name} {column.type.compile(engine.dialect)}"
                )
                default = getattr(column.default, "arg", None)
                if default is not None and not callable(default):
                    if not column.nullable:
                        ddl += " NOT NULL"
                    ddl += f" DEFAULT {default!r}"
                conn.execute(text(ddl))
