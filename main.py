"""
Main Application Entry Point

This module serves as the main entry point for the Telegram LLM Decision Engine.
It orchestrates all components including the Telegram handler, LLM processor,
decision engine, and various integration modules. The application provides
comprehensive error handling, graceful shutdown, and monitoring capabilities.

Author: Development Team
Version: 1.0.0
Date: 2025-09-16
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

# Core imports
from config.settings import get_config, AppConfig
from config.logging_config import setup_logging, get_logger, get_performance_logger
from core.telegram_handler import TelegramHandler, HandlerConfig
from core.llm_processor import LLMProcessor
from core.decision_engine import DecisionEngine
from core.state_manager import StateManager
from integrations.woocommerce_handler import WooCommerceHandler
from integrations.excel_handler import ExcelHandler
from storage.cache_manager import CacheManager
from utils.exceptions import ApplicationError, ConfigurationError
from utils.helpers import ensure_directories, cleanup_temp_files


class Application:
    """
    Main application class that orchestrates all components.
    
    This class is responsible for initializing all components, managing
    their lifecycle, handling graceful shutdown, and coordinating
    communication between different parts of the system.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the application with configuration.
        
        Args:
            config: Optional configuration override
        """
        # Load configuration
        self.config = config or get_config()
        
        # Setup logging first
        self.logger = None
        self.perf_logger = None
        self._setup_logging()
        
        # Initialize components
        self.state_manager: Optional[StateManager] = None
        self.cache_manager: Optional[CacheManager] = None
        self.llm_processor: Optional[LLMProcessor] = None
        self.decision_engine: Optional[DecisionEngine] = None
        self.telegram_handler: Optional[TelegramHandler] = None
        self.woocommerce_handler: Optional[WooCommerceHandler] = None
        self.excel_handler: Optional[ExcelHandler] = None
        
        # Application state
        self.is_running = False
        self.startup_time: Optional[datetime] = None
        self.shutdown_tasks: list = []
        
        self.logger.info(f"Application initialized - Version {self.config.app_version}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        try:
            # Convert logging config to dict format
            logging_config = {
                'level': self.config.logging.level,
                'format': self.config.logging.format,
                'date_format': self.config.logging.date_format,
                'log_to_file': self.config.logging.log_to_file,
                'log_file_path': self.config.logging.log_file_path,
                'max_file_size_mb': self.config.logging.max_file_size_mb,
                'backup_count': self.config.logging.backup_count,
                'log_to_console': self.config.logging.log_to_console,
                'console_color': self.config.logging.console_color,
                'module_levels': self.config.logging.module_levels,
                'enable_performance_logging': self.config.logging.enable_performance_logging,
                'slow_query_threshold_seconds': self.config.logging.slow_query_threshold_seconds,
                'enable_structured_logging': self.config.logging.enable_structured_logging,
                'log_context': self.config.logging.log_context
            }
            
            # Setup logging
            setup_logging(logging_config)
            
            # Get loggers
            self.logger = get_logger(__name__)
            self.perf_logger = get_performance_logger(__name__)
            
            self.logger.info("Logging system initialized successfully")
            
        except Exception as e:
            print(f"Failed to setup logging: {e}")
            sys.exit(1)
    
    async def initialize(self) -> None:
        """
        Initialize all application components.
        
        This method sets up all components in the correct order,
        handles dependencies, and performs health checks.
        
        Raises:
            ApplicationError: If initialization fails
        """
        try:
            self.logger.info("Starting application initialization...")
            
            # Ensure required directories exist
            await self._ensure_directories()
            
            # Initialize core components in dependency order
            await self._initialize_state_manager()
            await self._initialize_cache_manager()
            await self._initialize_llm_processor()
            await self._initialize_decision_engine()
            
            # Initialize integration handlers
            await self._initialize_woocommerce_handler()
            await self._initialize_excel_handler()
            
            # Initialize Telegram handler (last, as it depends on others)
            await self._initialize_telegram_handler()
            
            # Perform health checks
            await self._perform_health_checks()
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            self.startup_time = datetime.now(timezone.utc)
            self.logger.info("Application initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Application initialization failed: {e}")
            await self.shutdown()
            raise ApplicationError(f"Initialization failed: {e}")
    
    async def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        try:
            directories = [
                self.config.data_dir,
                self.config.downloads_dir,
                self.config.processed_dir,
                self.config.logs_dir,
                self.config.cache_dir
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Directory ensured: {directory}")
            
            self.logger.info("All required directories verified")
            
        except Exception as e:
            raise ApplicationError(f"Failed to create directories: {e}")
    
    async def _initialize_state_manager(self) -> None:
        """Initialize state management system."""
        try:
            self.state_manager = StateManager(
                database_url=self.config.database.database_url,
                redis_url=self.config.database.redis_url
            )
            await self.state_manager.initialize()
            
            # Register for cleanup
            self.shutdown_tasks.append(self.state_manager.close)
            
            self.logger.info("State manager initialized")
            
        except Exception as e:
            raise ApplicationError(f"State manager initialization failed: {e}")
    
    async def _initialize_cache_manager(self) -> None:
        """Initialize caching system."""
        try:
            self.cache_manager = CacheManager(
                redis_url=self.config.database.redis_url,
                ttl_seconds=self.config.database.cache_ttl_seconds,
                max_size_mb=self.config.database.max_cache_size_mb
            )
            await self.cache_manager.initialize()
            
            # Register for cleanup
            self.shutdown_tasks.append(self.cache_manager.close)
            
            self.logger.info("Cache manager initialized")
            
        except Exception as e:
            raise ApplicationError(f"Cache manager initialization failed: {e}")
    
    async def _initialize_llm_processor(self) -> None:
        """Initialize LLM processing system."""
        try:
            self.llm_processor = LLMProcessor()
            
            # Verify at least one provider is available
            available_providers = self.llm_processor.get_available_providers()
            if not available_providers:
                raise ConfigurationError("No LLM providers configured")
            
            self.logger.info(f"LLM processor initialized with providers: {available_providers}")
            
        except Exception as e:
            raise ApplicationError(f"LLM processor initialization failed: {e}")
    
    async def _initialize_decision_engine(self) -> None:
        """Initialize decision making engine."""
        try:
            self.decision_engine = DecisionEngine(
                llm_processor=self.llm_processor,
                state_manager=self.state_manager
            )
            await self.decision_engine.initialize()
            
            self.logger.info("Decision engine initialized")
            
        except Exception as e:
            raise ApplicationError(f"Decision engine initialization failed: {e}")
    
    async def _initialize_woocommerce_handler(self) -> None:
        """Initialize WooCommerce integration."""
        try:
            # Only initialize if configured
            if (self.config.woocommerce.store_url and 
                self.config.woocommerce.consumer_key and 
                self.config.woocommerce.consumer_secret):
                
                self.woocommerce_handler = WooCommerceHandler(
                    store_url=self.config.woocommerce.store_url,
                    consumer_key=self.config.woocommerce.consumer_key,
                    consumer_secret=self.config.woocommerce.consumer_secret
                )
                await self.woocommerce_handler.initialize()
                
                self.logger.info("WooCommerce handler initialized")
            else:
                self.logger.warning("WooCommerce not configured - handler skipped")
                
        except Exception as e:
            # WooCommerce is optional, log warning but don't fail
            self.logger.warning(f"WooCommerce handler initialization failed: {e}")
    
    async def _initialize_excel_handler(self) -> None:
        """Initialize Excel integration."""
        try:
            # Only initialize if configured
            if (self.config.excel.client_id and 
                self.config.excel.client_secret and 
                self.config.excel.tenant_id):
                
                self.excel_handler = ExcelHandler(
                    client_id=self.config.excel.client_id,
                    client_secret=self.config.excel.client_secret,
                    tenant_id=self.config.excel.tenant_id
                )
                await self.excel_handler.initialize()
                
                self.logger.info("Excel handler initialized")
            else:
                self.logger.warning("Excel not configured - handler skipped")
                
        except Exception as e:
            # Excel is optional, log warning but don't fail
            self.logger.warning(f"Excel handler initialization failed: {e}")
    
    async def _initialize_telegram_handler(self) -> None:
        """Initialize Telegram bot handler."""
        try:
            # Create handler configuration
            handler_config = HandlerConfig(
                max_file_size_mb=self.config.telegram.max_file_size_mb,
                session_timeout_minutes=self.config.security.session_timeout_minutes,
                max_messages_per_minute=self.config.security.api_rate_limit,
                enable_typing_indicator=True,
                enable_progress_updates=True
            )
            
            # Initialize handler
            self.telegram_handler = TelegramHandler(
                config=handler_config,
                state_manager=self.state_manager
            )
            
            # Inject dependencies
            self.telegram_handler.llm_processor = self.llm_processor
            self.telegram_handler.decision_engine = self.decision_engine
            self.telegram_handler.woocommerce_handler = self.woocommerce_handler
            self.telegram_handler.excel_handler = self.excel_handler
            
            await self.telegram_handler.initialize()
            
            # Register for cleanup
            self.shutdown_tasks.append(self.telegram_handler.stop)
            
            self.logger.info("Telegram handler initialized")
            
        except Exception as e:
            raise ApplicationError(f"Telegram handler initialization failed: {e}")
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all components."""
        try:
            health_results = {}
            
            # Check state manager
            if self.state_manager:
                health_results['state_manager'] = await self.state_manager.health_check()
            
            # Check cache manager
            if self.cache_manager:
                health_results['cache_manager'] = await self.cache_manager.health_check()
            
            # Check LLM processor
            if self.llm_processor:
                providers = self.llm_processor.get_available_providers()
                health_results['llm_processor'] = {
                    'status': 'healthy' if providers else 'degraded',
                    'providers': providers
                }
            
            # Check integrations
            if self.woocommerce_handler:
                health_results['woocommerce'] = await self.woocommerce_handler.health_check()
            
            if self.excel_handler:
                health_results['excel'] = await self.excel_handler.health_check()
            
            # Log health check results
            healthy_components = sum(1 for result in health_results.values() 
                                   if result.get('status') == 'healthy')
            total_components = len(health_results)
            
            self.logger.info(
                f"Health check completed: {healthy_components}/{total_components} "
                f"components healthy"
            )
            
            if healthy_components < total_components:
                self.logger.warning("Some components are not healthy")
                for component, result in health_results.items():
                    if result.get('status') != 'healthy':
                        self.logger.warning(f"{component}: {result}")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            # Don't fail startup for health check issues
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        # Handle common shutdown signals
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Docker stop
        
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)  # Terminal hangup
        
        self.logger.debug("Signal handlers configured")
    
    @perf_logger.log_function_performance("application_startup")
    async def run(self) -> None:
        """
        Run the application.
        
        This method starts all components and keeps the application running
        until a shutdown signal is received.
        """
        try:
            # Initialize components
            await self.initialize()
            
            # Mark as running
            self.is_running = True
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Start the Telegram bot
            self.logger.info("Starting Telegram bot...")
            await self.telegram_handler.start_polling()
            
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        try:
            # Cleanup task
            asyncio.create_task(self._periodic_cleanup())
            
            # Health monitoring task
            if self.config.enable_health_check:
                asyncio.create_task(self._periodic_health_check())
            
            # Metrics collection task
            if self.config.enable_metrics:
                asyncio.create_task(self._periodic_metrics_collection())
            
            self.logger.info("Background tasks started")
            
        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}")
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of temporary files and data."""
        cleanup_interval = self.config.cleanup_interval_hours * 3600
        
        while self.is_running:
            try:
                await asyncio.sleep(cleanup_interval)
                
                if not self.is_running:
                    break
                
                self.logger.info("Starting periodic cleanup...")
                
                # Cleanup temporary files
                await cleanup_temp_files(self.config.cache_dir, hours=24)
                await cleanup_temp_files(self.config.downloads_dir, hours=48)
                
                # Cleanup old logs if needed
                if self.config.logging.backup_count > 0:
                    log_dir = Path(self.config.logs_dir)
                    if log_dir.exists():
                        log_files = sorted(log_dir.glob("*.log.*"))
                        if len(log_files) > self.config.logging.backup_count:
                            for old_log in log_files[:-self.config.logging.backup_count]:
                                old_log.unlink()
                                self.logger.debug(f"Deleted old log: {old_log}")
                
                self.logger.info("Periodic cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Periodic cleanup error: {e}")
    
    async def _periodic_health_check(self) -> None:
        """Periodic health monitoring."""
        check_interval = 300  # 5 minutes
        
        while self.is_running:
            try:
                await asyncio.sleep(check_interval)
                
                if not self.is_running:
                    break
                
                await self._perform_health_checks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
    
    async def _periodic_metrics_collection(self) -> None:
        """Periodic metrics collection and reporting."""
        metrics_interval = 600  # 10 minutes
        
        while self.is_running:
            try:
                await asyncio.sleep(metrics_interval)
                
                if not self.is_running:
                    break
                
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Log metrics
                self.logger.info(f"System metrics: {metrics}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        metrics = {
            'uptime_seconds': (datetime.now(timezone.utc) - self.startup_time).total_seconds() if self.startup_time else 0,
            'memory_usage_mb': 0,  # Would implement with psutil
            'active_sessions': 0,
            'processed_tasks': 0,
            'error_count': 0
        }
        
        try:
            # Get metrics from components
            if self.state_manager:
                state_metrics = await self.state_manager.get_metrics()
                metrics.update(state_metrics)
            
            if self.llm_processor:
                llm_metrics = self.llm_processor.get_usage_stats()
                metrics['llm_usage'] = llm_metrics
            
            if self.telegram_handler:
                tg_metrics = await self.telegram_handler.get_metrics()
                metrics.update(tg_metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the application.
        
        This method stops all components in reverse order of initialization,
        cleans up resources, and ensures data integrity.
        """
        if not self.is_running:
            return
        
        try:
            self.logger.info("Initiating graceful shutdown...")
            self.is_running = False
            
            # Stop components in reverse order
            for shutdown_task in reversed(self.shutdown_tasks):
                try:
                    if asyncio.iscoroutinefunction(shutdown_task):
                        await shutdown_task()
                    else:
                        shutdown_task()
                except Exception as e:
                    self.logger.error(f"Error during shutdown task: {e}")
            
            # Final cleanup
            await self._final_cleanup()
            
            # Calculate uptime
            if self.startup_time:
                uptime = datetime.now(timezone.utc) - self.startup_time
                self.logger.info(f"Application shutdown complete. Uptime: {uptime}")
            else:
                self.logger.info("Application shutdown complete")
                
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def _final_cleanup(self) -> None:
        """Perform final cleanup tasks."""
        try:
            # Save any pending state
            if self.state_manager:
                await self.state_manager.flush_all()
            
            # Clear cache if needed
            if self.cache_manager:
                await self.cache_manager.clear_expired()
            
            self.logger.debug("Final cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Final cleanup error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current application status.
        
        Returns:
            Dictionary containing application status information
        """
        status = {
            'running': self.is_running,
            'version': self.config.app_version,
            'environment': self.config.environment,
            'startup_time': self.startup_time.isoformat() if self.startup_time else None,
            'uptime_seconds': (datetime.now(timezone.utc) - self.startup_time).total_seconds() if self.startup_time else 0,
            'components': {}
        }
        
        # Component status
        status['components']['state_manager'] = bool(self.state_manager)
        status['components']['cache_manager'] = bool(self.cache_manager)
        status['components']['llm_processor'] = bool(self.llm_processor)
        status['components']['decision_engine'] = bool(self.decision_engine)
        status['components']['telegram_handler'] = bool(self.telegram_handler)
        status['components']['woocommerce_handler'] = bool(self.woocommerce_handler)
        status['components']['excel_handler'] = bool(self.excel_handler)
        
        return status


async def main():
    """Main entry point for the application."""
    try:
        # Create and run application
        app = Application()
        await app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application failed: {e}")
        sys.exit(1)


def run_development():
    """Run application in development mode with debug settings."""
    import os
    os.environ['ENVIRONMENT'] = 'development'
    os.environ['DEBUG_MODE'] = 'true'
    
    asyncio.run(main())


def run_production():
    """Run application in production mode."""
    import os
    os.environ['ENVIRONMENT'] = 'production'
    os.environ['DEBUG_MODE'] = 'false'
    
    asyncio.run(main())


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "dev":
            run_development()
        elif sys.argv[1] == "prod":
            run_production()
        else:
            print("Usage: python main.py [dev|prod]")
            sys.exit(1)
    else:
        # Default to development mode
        run_development()


# Export for use as a module
__all__ = ['Application', 'main', 'run_development', 'run_production']