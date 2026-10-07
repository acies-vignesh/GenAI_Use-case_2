"""Alembic environment: points migrations at our models and the app database."""
from alembic import context

from app.config import settings
from app.persistence.database import make_engine
from app.persistence.models import Base

config = context.config
url = config.get_main_option("sqlalchemy.url") or settings.app_db_url
target_metadata = Base.metadata


def run_offline() -> None:
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    with make_engine(url).connect() as connection:
        # render_as_batch: SQLite can't ALTER most things in place; batch mode rebuilds the table safely
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
