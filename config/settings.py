"""
Configuration Settings Module

Provides configuration management for the Telegram LLM Decision Engine
using Pydantic v2 and pydantic-settings.

Author: GG
Version: 0.1.0
Date: 2025-09-16
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class SecurityConfig(BaseSettings):
    """
    Security-related configuration settings.
    """

    model_config = SettingsConfigDict(env_prefix='SECURITY_')

    encryption_key: Optional[str] = Field(default=None, description="Fernet encryption key for sensitive data")
    api_rate_limit: int = Field(default=60, description="API requests per minute limit")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent API requests")
    session_timeout_minutes: int = Field(default=30, description="User session timeout in minutes")
    token_expiry_hours: int = Field(default=24, description="API token expiry in hours")

    @field_validator('encryption_key', mode='before')
    @classmethod
    def validate_encryption_key(cls, v):
        if v is None:
            logger.warning("No encryption key provided, generating new key")
            return Fernet.generate_key().decode()
        return v


class TelegramConfig(BaseSettings):
    """
    Telegram Bot API configuration settings.
    """

    model_config = SettingsConfigDict(env_prefix='TELEGRAM_')

    bot_token: str = Field(..., description="Telegram Bot API token")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for Telegram updates")
    webhook_secret: Optional[str] = Field(default=None, description="Webhook secret token")

    max_file_size_mb: int = Field(default=50, description="Maximum file size for uploads in MB")
    allowed_file_types: List[str] = Field(
        default=['jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov', 'pdf', 'txt', 'doc', 'docx'],
        description="Allowed file extensions for uploads"
    )

    max_message_length: int = Field(default=4096, description="Maximum message length")
    typing_delay_seconds: float = Field(default=0.5, description="Typing indicator delay")

    enable_group_chat: bool = Field(default=False, description="Enable bot in group chats")
    admin_user_ids: List[int] = Field(default_factory=list, description="Administrator user IDs")

    @field_validator('bot_token')
    @classmethod
    def validate_bot_token(cls, v):
        if not v or ':' not in v:
            raise ValueError("Invalid Telegram bot token format")
        return v

    @field_validator('max_file_size_mb')
    @classmethod
    def validate_file_size(cls, v):
        if v <= 0 or v > 100:
            raise ValueError("File size must be between 1-100 MB")
        return v


class LLMConfig(BaseSettings):
    """
    Large Language Model configuration settings.
    """

    model_config = SettingsConfigDict(env_prefix='LLM_')

    primary_provider: str = Field(default="openai", description="Primary LLM provider")
    fallback_providers: List[str] = Field(default=["anthropic"], description="Fallback LLM providers")

    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="Default OpenAI model")
    openai_temperature: float = Field(default=0.7, description="OpenAI temperature setting")
    openai_max_tokens: int = Field(default=4096, description="OpenAI max tokens")
    openai_timeout_seconds: int = Field(default=60, description="OpenAI request timeout")

    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", description="Default Anthropic model")
    anthropic_temperature: float = Field(default=0.7, description="Anthropic temperature setting")
    anthropic_max_tokens: int = Field(default=4096, description="Anthropic max tokens")

    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay_seconds: float = Field(default=1.0, description="Delay between retries")
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")

    monthly_token_limit: Optional[int] = Field(default=1000000, description="Monthly token usage limit")
    cost_per_token: float = Field(default=0.00001, description="Cost per token for budgeting")

    @field_validator('primary_provider')
    @classmethod
    def validate_provider(cls, v):
        allowed_providers = ['openai', 'anthropic', 'local']
        if v not in allowed_providers:
            raise ValueError(f"Provider must be one of: {allowed_providers}")
        return v

    @field_validator('openai_temperature', 'anthropic_temperature')
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class WooCommerceConfig(BaseSettings):
    """
    WooCommerce API configuration settings.
    """

    model_config = SettingsConfigDict(env_prefix='WOOCOMMERCE_')

    store_url: Optional[str] = Field(default=None, description="WooCommerce store URL")
    consumer_key: Optional[str] = Field(default=None, description="WooCommerce API consumer key")
    consumer_secret: Optional[str] = Field(default=None, description="WooCommerce API consumer secret")
    api_version: str = Field(default="wc/v3", description="WooCommerce API version")

    timeout_seconds: int = Field(default=30, description="API request timeout")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    enable_webhook: bool = Field(default=False, description="Enable WooCommerce webhooks")

    default_status: str = Field(default="draft", description="Default product status")
    auto_generate_sku: bool = Field(default=True, description="Auto-generate product SKU")
    default_category_id: Optional[int] = Field(default=None, description="Default product category")

    max_images_per_product: int = Field(default=10, description="Maximum images per product")
    image_quality: int = Field(default=85, description="Image quality for uploads (1-100)")

    @field_validator('store_url')
    @classmethod
    def validate_store_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("Store URL must include protocol (http:// or https://)")
        return v

    @field_validator('default_status')
    @classmethod
    def validate_status(cls, v):
        allowed_statuses = ['draft', 'pending', 'private', 'publish']
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {allowed_statuses}")
        return v


class ExcelConfig(BaseSettings):
    """
    Microsoft Excel Online/Graph API configuration settings.
    """

    model_config = SettingsConfigDict(env_prefix='EXCEL_')

    client_id: Optional[str] = Field(default=None, description="Azure AD application client ID")
    client_secret: Optional[str] = Field(default=None, description="Azure AD application client secret")
    tenant_id: Optional[str] = Field(default=None, description="Azure AD tenant ID")

    workbook_id: Optional[str] = Field(default=None, description="Target Excel workbook ID")
    worksheet_name: str = Field(default="Data", description="Default worksheet name")
    start_row: int = Field(default=1, description="Starting row for data insertion")
    start_column: str = Field(default="A", description="Starting column for data insertion")

    api_base_url: str = Field(default="https://graph.microsoft.com/v1.0", description="Microsoft Graph API base URL")
    timeout_seconds: int = Field(default=60, description="API request timeout")

    batch_size: int = Field(default=100, description="Batch size for bulk operations")
    auto_format: bool = Field(default=True, description="Auto-format Excel data")
    backup_enabled: bool = Field(default=True, description="Enable automatic backups")

    @field_validator('start_column')
    @classmethod
    def validate_column(cls, v):
        if not v.isalpha() or not v.isupper():
            raise ValueError("Column must be uppercase letter(s)")
        return v


class DatabaseConfig(BaseSettings):
    """
    Database configuration settings.
    """

    model_config = SettingsConfigDict(env_prefix='DATABASE_')

    database_url: str = Field(default="sqlite+aiosqlite:///data/app.db", description="Database connection URL")
    echo_sql: bool = Field(default=False, description="Echo SQL queries to console")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum connection overflow")

    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    max_cache_size_mb: int = Field(default=100, description="Maximum cache size in MB")

    backup_enabled: bool = Field(default=True, description="Enable automatic backups")
    backup_interval_hours: int = Field(default=24, description="Backup interval in hours")
    max_backup_files: int = Field(default=7, description="Maximum backup files to keep")


@dataclass
class LoggingConfig:
    """
    Logging configuration settings.
    """

    level: str = field(default="INFO")
    format: str = field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format: str = field(default="%Y-%m-%d %H:%M:%S")

    log_to_file: bool = field(default=True)
    log_file_path: str = field(default="data/logs/app.log")
    max_file_size_mb: int = field(default=10)
    backup_count: int = field(default=5)

    log_to_console: bool = field(default=True)
    console_color: bool = field(default=True)

    module_levels: Dict[str, str] = field(default_factory=lambda: {
        'telegram': 'INFO',
        'llm': 'DEBUG',
        'woocommerce': 'INFO',
        'excel': 'INFO',
        'decision_engine': 'DEBUG'
    })

    enable_performance_logging: bool = field(default=True)
    slow_query_threshold_seconds: float = field(default=1.0)

    enable_structured_logging: bool = field(default=True)
    log_context: bool = field(default=True)


class AppConfig(BaseSettings):
    """
    Main application configuration class.
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
    )

    app_name: str = Field(default="Telegram LLM Decision Engine", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Deployment environment")
    debug_mode: bool = Field(default=False, description="Enable debug mode")

    data_dir: Path = Field(default=Path("data"), description="Base data directory")
    downloads_dir: Path = Field(default=Path("data/downloads"), description="Downloads directory")
    processed_dir: Path = Field(default=Path("data/processed"), description="Processed files directory")
    logs_dir: Path = Field(default=Path("data/logs"), description="Logs directory")
    cache_dir: Path = Field(default=Path("data/cache"), description="Cache directory")

    max_concurrent_users: int = Field(default=100, description="Maximum concurrent users")
    request_timeout_seconds: int = Field(default=30, description="Global request timeout")
    cleanup_interval_hours: int = Field(default=24, description="Cleanup interval for temporary files")

    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_health_check: bool = Field(default=True, description="Enable health check endpoint")
    enable_admin_panel: bool = Field(default=False, description="Enable admin panel")

    security: SecurityConfig = Field(default_factory=SecurityConfig)
    #telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    woocommerce: WooCommerceConfig = Field(default_factory=WooCommerceConfig)
    excel: ExcelConfig = Field(default_factory=ExcelConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        self._create_directories()
        self._validate_configuration()
        logger.info(f"Configuration loaded for environment: {self.environment}")

    def _create_directories(self) -> None:
        directories = [
            self.data_dir,
            self.downloads_dir,
            self.processed_dir,
            self.logs_dir,
            self.cache_dir,
        ]
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory ensured: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise

    def _validate_configuration(self) -> None:
        if not self.telegram.bot_token:
            raise ValueError("Telegram bot token is required")

        if self.llm.primary_provider == "openai" and not self.llm.openai_api_key:
            logger.warning("OpenAI API key not configured")

        if self.llm.primary_provider == "anthropic" and not self.llm.anthropic_api_key:
            logger.warning("Anthropic API key not configured")

        if self.woocommerce.store_url and not (self.woocommerce.consumer_key and self.woocommerce.consumer_secret):
            logger.warning("WooCommerce credentials incomplete")

        if self.excel.client_id and not (self.excel.client_secret and self.excel.tenant_id):
            logger.warning("Excel/Microsoft Graph credentials incomplete")

        logger.info("Configuration validation completed")

    def get_api_key(self, service: str, encrypted: bool = True) -> Optional[str]:
        service_keys = {
            'telegram': self.telegram.bot_token,
            'openai': self.llm.openai_api_key,
            'anthropic': self.llm.anthropic_api_key,
            'woocommerce_key': self.woocommerce.consumer_key,
            'woocommerce_secret': self.woocommerce.consumer_secret,
            'excel_client': self.excel.client_id,
            'excel_secret': self.excel.client_secret,
        }
        if service not in service_keys:
            raise ValueError(f"Unknown service: {service}")
        key = service_keys.get(service)
        if not key:
            logger.warning(f"No API key configured for service: {service}")
            return None
        if encrypted and self.security.encryption_key:
            try:
                if key.startswith('gAAAAA'):  # Fernet encrypted data signature
                    cipher = Fernet(self.security.encryption_key.encode())
                    key = cipher.decrypt(key.encode()).decode()
            except Exception as e:
                logger.error(f"Failed to decrypt API key for {service}: {e}")
                raise
        return key

    def update_setting(self, section: str, key: str, value: Any) -> bool:
        try:
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
                    logger.info(f"Updated {section}.{key} = {value}")
                    return True
                else:
                    logger.error(f"Invalid key '{key}' for section '{section}'")
            else:
                logger.error(f"Invalid section '{section}'")
            return False
        except Exception as e:
            logger.error(f"Failed to update setting {section}.{key}: {e}")
            return False

    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        config_dict = {}
        sections = ['security', 'telegram', 'llm', 'woocommerce', 'excel', 'database']
        for section_name in sections:
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                section_dict = {}
                if hasattr(section, 'model_dump'):
                    section_dict = section.model_dump()
                else:
                    section_dict = section.__dict__
                if not include_secrets:
                    secret_keys = ['api_key', 'token', 'secret', 'password', 'key']
                    for key in list(section_dict.keys()):
                        if any(secret_key in key.lower() for secret_key in secret_keys):
                            section_dict[key] = "***HIDDEN***"
                config_dict[section_name] = section_dict
        return config_dict


config = AppConfig()


def get_config() -> AppConfig:
    """
    Get the global configuration instance.

    Returns:
        Global AppConfig instance
    """
    return config


def reload_config() -> AppConfig:
    """
    Reload configuration from environment and files.

    Returns:
        Reloaded AppConfig instance
    """
    global config
    config = AppConfig()
    return config


__all__ = [
    'AppConfig',
    'SecurityConfig',
    'TelegramConfig',
    'LLMConfig',
    'WooCommerceConfig',
    'ExcelConfig',
    'DatabaseConfig',
    'LoggingConfig',
    'get_config',
    'reload_config',
    'config'
]
