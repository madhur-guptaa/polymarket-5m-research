"""
polaris/core/config.py

Single source of truth for configuration, built on pydantic-settings.

Every setting is read from the environment (and a local .env), typed, and
validated at construction. Invalid config fails loudly at startup instead of
surfacing as a mysterious runtime error.

WHY pydantic-settings (and where it stops):
    Config is loaded once at boot, so validation cost is irrelevant and the
    safety is pure upside — env parsing, type coercion, and range checks that
    we used to hand-roll now come for free. This is exactly the layer Pydantic
    belongs in. It is deliberately kept OFF the trade hot path: ingested trades
    are plain tuples (see ingest/), never Pydantic models.

LAYOUT:
    Settings                      root, composed of the sections below
    ├── pg:      PostgresSettings  connection + pool sizing
    ├── binance: BinanceSettings   symbols + websocket stream
    └── ingest:  IngestSettings    batch/flush tuning for the live sink

Each section is its own BaseSettings reading flat env vars (POSTGRES_*,
BINANCE_*, INGEST_*), so the names match the existing .env.example. New
sections (model, trading, api) get added here as those components land.
"""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Shared env-file config: read .env, ignore unrelated keys, case-insensitive.
_ENV = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
    extra="ignore",
)


class PostgresSettings(BaseSettings):
    """PostgreSQL connection + pool sizing.

    A full ``POSTGRES_DSN`` wins if set; otherwise the DSN is composed from the
    individual ``POSTGRES_*`` parts (matching the old behaviour).
    """

    model_config = _ENV

    dsn_override: str | None = Field(default=None, alias="POSTGRES_DSN")
    user: str = Field(default="polaris", alias="POSTGRES_USER")
    password: str = Field(default="polaris", alias="POSTGRES_PASSWORD")
    host: str = Field(default="localhost", alias="POSTGRES_HOST")
    port: int = Field(default=5432, alias="POSTGRES_PORT")
    database: str = Field(default="polaris", alias="POSTGRES_DB")

    pool_min: int = Field(default=2, ge=1, alias="PG_POOL_MIN_CONN")
    pool_max: int = Field(default=10, ge=1, alias="PG_POOL_MAX_CONN")

    @field_validator("pool_max")
    @classmethod
    def _pool_max_ge_min(cls, v: int, info) -> int:
        pool_min = info.data.get("pool_min", 1)
        if v < pool_min:
            raise ValueError(f"PG_POOL_MAX_CONN ({v}) must be >= PG_POOL_MIN_CONN ({pool_min})")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dsn(self) -> str:
        """libpq connection string used by psycopg."""
        if self.dsn_override:
            return self.dsn_override
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def safe_dsn(self) -> str:
        """DSN with the password masked — safe to log."""
        return re.sub(r"(?<=:)[^@:]+(?=@)", "***", self.dsn)


class BinanceSettings(BaseSettings):
    """Binance symbol universe and websocket stream parameters."""

    model_config = _ENV

    # Comma-separated in the env (e.g. "btcusdt,ethusdt"); parsed by `symbols`.
    # Kept as a raw string so pydantic-settings does not try to JSON-decode it.
    symbols_raw: str = Field(default="btcusdt,ethusdt", alias="BINANCE_SYMBOLS")

    ws_base_url: str = Field(
        default="wss://stream.binance.com:9443/stream",
        alias="BINANCE_WS_BASE_URL",
    )
    reconnect_delay_sec: float = Field(default=3.0, gt=0, alias="BINANCE_RECONNECT_DELAY_SEC")
    ping_interval_sec: float = Field(default=20.0, gt=0, alias="BINANCE_PING_INTERVAL_SEC")
    ping_timeout_sec: float = Field(default=20.0, gt=0, alias="BINANCE_PING_TIMEOUT_SEC")
    max_frame_bytes: int = Field(default=1 << 20, gt=0, alias="BINANCE_MAX_FRAME_BYTES")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def symbols(self) -> list[str]:
        """Configured symbols, upper-cased and de-duplicated (order preserved)."""
        seen: dict[str, None] = {}
        for tok in self.symbols_raw.split(","):
            s = tok.strip().upper()
            if s:
                seen.setdefault(s, None)
        if not seen:
            raise ValueError("BINANCE_SYMBOLS must list at least one symbol")
        return list(seen)

    @property
    def combined_stream_url(self) -> str:
        """Binance combined-stream URL: <base>?streams=btcusdt@trade/ethusdt@trade."""
        streams = "/".join(f"{s.lower()}@trade" for s in self.symbols)
        return f"{self.ws_base_url}?streams={streams}"


class IngestSettings(BaseSettings):
    """Tuning for the live WS -> PostgreSQL sink."""

    model_config = _ENV

    # Flush when the buffer hits this many rows...
    batch_size: int = Field(default=1000, ge=1, alias="INGEST_BATCH_SIZE")
    # ...or when this many seconds have elapsed since the last flush.
    flush_interval_sec: float = Field(default=1.0, gt=0, alias="INGEST_FLUSH_INTERVAL_SEC")
    # Apply schema.sql on startup (safe/idempotent). Turn off in prod if desired.
    ensure_schema: bool = Field(default=True, alias="INGEST_ENSURE_SCHEMA")


class AggregateSettings(BaseSettings):
    """Tuning for the trades -> 1s bars -> window observations aggregator."""

    model_config = _ENV

    # Trading window length. 300 = 5 minutes. Must be consistent everywhere.
    window_sec: int = Field(default=300, gt=0, alias="WINDOW_SEC")
    # Wait this long before aggregating a time range, so late trades land first.
    safe_lag_sec: int = Field(default=2, ge=0, alias="AGGREGATE_SAFE_LAG_SEC")
    # How often to poll for new data when caught up.
    poll_interval_sec: float = Field(default=1.0, gt=0, alias="AGGREGATE_POLL_INTERVAL_SEC")
    # Max wall-clock span processed per cycle (bounds catch-up query size after
    # a backfill). Rounded down to a whole number of windows at use.
    max_chunk_sec: int = Field(default=3600, gt=0, alias="AGGREGATE_MAX_CHUNK_SEC")


class PolymarketSettings(BaseSettings):
    """Polymarket API access (Gamma discovery + CLOB prices)."""

    model_config = _ENV

    gamma_api_url: str = Field(
        default="https://gamma-api.polymarket.com/events", alias="POLY_GAMMA_URL")
    clob_api_url: str = Field(
        default="https://clob.polymarket.com", alias="POLY_CLOB_URL")
    # Trade-level history (individual fills, second resolution, ~60d back).
    # /prices-history on the CLOB only serves 1-minute resolution, which is
    # what invalidated the original backtest prices.
    data_api_url: str = Field(
        default="https://data-api.polymarket.com", alias="POLY_DATA_API_URL")
    timeout_sec: float = Field(default=15.0, gt=0, alias="POLY_TIMEOUT_SEC")

    # Sliding-window rate limits (published limits are 500/10s Gamma, 1000/10s
    # CLOB; we take ~90% as headroom against 429 bans).
    gamma_max_req: int = Field(default=450, gt=0, alias="POLY_GAMMA_MAX_REQ")
    gamma_window_sec: float = Field(default=10.0, gt=0, alias="POLY_GAMMA_WINDOW_SEC")
    clob_max_req: int = Field(default=900, gt=0, alias="POLY_CLOB_MAX_REQ")
    clob_window_sec: float = Field(default=10.0, gt=0, alias="POLY_CLOB_WINDOW_SEC")
    # data-api throttles MUCH harder than the documented CLOB budget: a trade
    # backfill at concurrency 10 drew 202 429s and silently lost 2/3 of its
    # windows. Measured-conservative rather than doc-derived.
    data_max_req: int = Field(default=40, gt=0, alias="POLY_DATA_MAX_REQ")
    data_window_sec: float = Field(default=10.0, gt=0, alias="POLY_DATA_WINDOW_SEC")

    backfill_concurrency: int = Field(default=10, ge=1, alias="POLY_BACKFILL_CONCURRENCY")

    # book-collect: how often to snapshot the live CLOB book, and how many
    # levels per side to retain. 5s covers the 45-90s trade gate densely at
    # ~0.4 req/s -- negligible against the 90 req/s CLOB budget.
    book_interval_sec: float = Field(default=5.0, gt=0, alias="POLY_BOOK_INTERVAL_SEC")
    book_keep_levels: int = Field(default=10, ge=1, alias="POLY_BOOK_KEEP_LEVELS")
    # Seconds before a window boundary to prefetch (used by live discovery, later).
    prefetch_lead_sec: int = Field(default=30, ge=0, alias="POLY_PREFETCH_LEAD_SEC")


class ModelSettings(BaseSettings):
    """Kernel bandwidths + Beta-Binomial prior for the KDE model.

    Bandwidths are the OUTPUT of Step-1 calibration; the defaults here are
    reasonable starting points. `use_tod` toggles the 6th (time-of-day)
    dimension so calibration can compare 5D vs 6D block-CV Brier.
    """

    model_config = _ENV

    bw_delta: float = Field(default=5.0, gt=0, alias="BW_DELTA")
    bw_time: float = Field(default=15.0, gt=0, alias="BW_TIME")
    bw_momentum: float = Field(default=3.0, gt=0, alias="BW_MOMENTUM")
    bw_imbalance: float = Field(default=0.05, gt=0, alias="BW_IMBALANCE")
    bw_volatility: float = Field(default=0.3, gt=0, alias="BW_VOLATILITY")
    bw_tod: float = Field(default=0.4, gt=0, alias="BW_TOD")

    # Which estimator serves, backtests and calibrates by default. "kde_twap"
    # (model/bayesian/twap.py) is the TWAP-geometry KDE; "kde" is the original
    # 7-dimension Binance-microstructure KDE, kept as a baseline; "gbm" is the
    # LightGBM control. See model/factory.py.
    kind: str = Field(default="kde_twap", alias="MODEL_KIND")

    # TWAP-geometry KDE bandwidths (twap.py). Measured 2026-09-29: Optuna block-CV
    # on real settlements, post-2026-08-07, 60 trials per symbol, then the
    # geometric mean of the BTC and ETH optima (sharing one set cost BTC 0.0005
    # Brier and ETH nothing). twap_sofar and req come out effectively FLAT --
    # widths far beyond their data scale -- because the tuner turns them off:
    # spot + locked alone match all four dimensions to 5 decimals. They are kept
    # as dimensions (the GBM does use them) but at this sample size a product
    # kernel cannot. See RESEARCH_LOG session 28.
    bw_twap_sofar: float = Field(default=65.0, gt=0, alias="BW_TWAP_SOFAR")
    bw_req: float = Field(default=3000.0, gt=0, alias="BW_REQ")
    bw_locked: float = Field(default=0.157, gt=0, alias="BW_LOCKED")
    bw_spot: float = Field(default=1.0, gt=0, alias="BW_SPOT")

    prior_alpha: float = Field(default=2.0, gt=0, alias="PRIOR_ALPHA")
    prior_beta: float = Field(default=2.0, gt=0, alias="PRIOR_BETA")

    training_window_days: int = Field(default=14, ge=1, alias="TRAINING_WINDOW_DAYS")
    momentum_lookback_sec: int = Field(default=15, ge=1, alias="MOMENTUM_LOOKBACK_SEC")
    use_tod: bool = Field(default=True, alias="USE_TOD")
    # Serving: re-fit the live estimator from the DB when older than this.
    estimator_refresh_sec: float = Field(default=60.0, gt=0, alias="ESTIMATOR_REFRESH_SEC")


class TradingSettings(BaseSettings):
    """Signal gates for the trade / no-trade readout (not execution)."""

    model_config = _ENV

    min_edge: float = Field(default=0.03, ge=0, alias="MIN_EDGE")
    # Polymarket's crypto taker fee: fee per share = rate * p * (1 - p), paid in
    # USDC by the taker only (makers pay 0 and receive 20% of it as rebates).
    # Peaks at 1.75c per share at p = 0.5 -- comparable to every edge this project
    # has ever measured, and omitted from all P&L before 2026-10-07.
    taker_fee_rate: float = Field(default=0.07, ge=0, alias="TAKER_FEE_RATE")
    min_n_eff: float = Field(default=20.0, ge=0, alias="MIN_N_EFF")
    max_kelly_fraction: float = Field(default=0.05, gt=0, alias="MAX_KELLY_FRACTION")

    # Only act inside this elapsed-seconds window. The diagnostic (session 10)
    # showed the tradeable edge lives ~45-75s: earlier is too noisy, and past
    # ~90s the market is already informed (win rate collapses). Provisional,
    # in-sample BTC — tune once the label is Chainlink-aligned.
    trade_window_start: int = Field(default=45, ge=0, alias="TRADE_WINDOW_START")
    trade_window_end: int = Field(default=90, ge=1, alias="TRADE_WINDOW_END")


class Settings(BaseSettings):
    """Root configuration. Import via ``get_settings()``."""

    model_config = _ENV

    pg: PostgresSettings = Field(default_factory=PostgresSettings)
    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    aggregate: AggregateSettings = Field(default_factory=AggregateSettings)
    polymarket: PolymarketSettings = Field(default_factory=PolymarketSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once and cache them for the process lifetime."""
    return Settings()
